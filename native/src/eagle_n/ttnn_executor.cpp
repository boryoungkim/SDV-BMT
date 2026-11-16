// BOS SDK-specific native graph executor. C++ only; model constants/shape plan are loaded at prepare.
// Capturing/qualifying a model is a separate hardware gate; no ACTUAL qualification is fabricated here.
#include "executor_abi.hpp"
#include "../common/json.hpp"
#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/conv/conv2d/conv2d.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/data_movement/concat/concat.hpp>
#include <ttnn/operations/data_movement/pad/pad.hpp>
#include <ttnn/operations/data_movement/repeat/repeat.hpp>
#include <ttnn/operations/pool/generic/generic_pools.hpp>
#include <ttnn/operations/pool/upsample/upsample.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/operations/reduction/argmax/argmax.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/eltwise/binary/binary_composite.hpp>
#include <ttnn/operations/trace.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <vector>
namespace sdv::eagle {
using T=ttnn::Tensor;using S=ttnn::Shape;using DT=ttnn::DataType;using MC=ttnn::MemoryConfig;
const auto dram=ttnn::DRAM_MEMORY_CONFIG;
using A=std::array<uint32_t,4>;
inline int I(const Json& j,const char* k){return int(j.at(k).as_int());}
inline std::string readf(const std::string& p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot read "+p);return {std::istreambuf_iterator<char>(f),{}};}
inline A dims(const T&t){auto s=t.logical_shape();return {s[0],s[1],s[2],s[3]};}
inline T shape(const T&t,A s){return ttnn::reshape(t,S(s));}
inline T cut(const T&t,A lo,A hi){return ttnn::slice(t,lo,hi,A{1,1,1,1},dram);}
inline T tile(const T&t){return ttnn::to_layout(t,ttnn::TILE_LAYOUT);}
inline T row(const T&t){return ttnn::to_memory_config(ttnn::to_layout(t,ttnn::ROW_MAJOR_LAYOUT),dram);}
inline T join(std::vector<T> p,int axis){if(p.empty())throw std::runtime_error("empty join");while(p.size()>1){std::vector<T> n;for(size_t i=0;i<p.size();i+=2)n.push_back(i+1<p.size()?ttnn::concat(std::vector<T>{p[i],p[i+1]},axis,dram):p[i]);p=std::move(n);}return p[0];}
struct F {T t;uint32_t h,w,c;};
// R18 "tuned" graph (frozen 2026-11-15, caontd/eagle_n_porting/deeplab_r18/CHOT_R18_LATENCY_20261115.md):
// same operators as the validated Plan C graph, with per-layer Conv2dConfig, a fused final resize + tile ArgMax,
// direct TILE residual add, 32-row MaxPool chunks and row-page input. Labels must equal the validated output.
inline bool has(const std::string&s,const char*p){return s.find(p)!=std::string::npos;}
inline bool tile_ready(const F&f){if(f.t.layout()!=ttnn::TILE_LAYOUT)return false;auto d=dims(f.t);return d[0]==1&&d[1]==1&&d[2]==f.h*f.w&&d[3]==f.c;}
// Mirror of the SDK bilinear autoshard rule (largest core count <=20 dividing the input rows).
inline uint32_t upsample_rows_per_core(uint32_t rows_in){uint32_t cores=std::min<uint32_t>(20,rows_in);while(rows_in%cores)--cores;return rows_in/cores;}
// Input-row chunks [start,end) with a 1-row halo; each chunk's halo-inclusive rows use <=per_core rows per core.
inline std::vector<std::pair<uint32_t,uint32_t>> plan_chunks(uint32_t total,uint32_t per_core){std::vector<std::pair<uint32_t,uint32_t>> b;uint32_t start=0;
 while(start<total){uint32_t end=std::min(total,start+20*per_core);for(;end>start+1;--end){uint32_t rin=(end-start)+(start>0)+(end<total);if(upsample_rows_per_core(rin)<=per_core)break;}b.push_back({start,end});start=end;}return b;}
constexpr uint32_t kClasses=19;
inline T rows(const F&f){auto src=f.t;auto d=dims(src);if(src.layout()==ttnn::TILE_LAYOUT)src=shape(src,{1,1,f.h*f.w,d[3]});auto t=row(src);auto c=dims(t)[3];t=shape(t,{1,f.h,f.w,c});return c==f.c?t:cut(t,{0,0,0,0},{1,f.h,f.w,f.c});}
inline T tensor(const std::vector<float>& v,A s,DT dtype){auto spec=ttnn::TensorSpec(S(s),tt::tt_metal::TensorLayout(dtype,tt::tt_metal::PageConfig(ttnn::ROW_MAJOR_LAYOUT),dram));
 if(dtype==DT::BFLOAT16){std::vector<bfloat16>b; b.reserve(v.size());for(float x:v)b.emplace_back(x);return T::from_vector(std::move(b),spec);}return T::from_vector(v,spec);}
inline T cast_rows(const T&src,DT dtype){
 if(src.dtype()==dtype)return src;auto d=dims(src);uint32_t pixels=d[1]*d[2],c=d[3];auto flat=shape(row(src),{1,1,pixels,c});std::vector<T> pp;
 for(uint32_t p=0;p<pixels;p+=32){uint32_t pe=std::min(p+32,pixels);std::vector<T> cc;
  for(uint32_t ch=0;ch<c;ch+=32){uint32_t ce=std::min(ch+32,c);auto part=cut(flat,{0,0,p,ch},{1,1,pe,ce});auto val=row(ttnn::typecast(tile(part),dtype,dram));cc.push_back(shape(val,{1,1,pe-p,ce-ch}));}pp.push_back(join(std::move(cc),3));}
 return shape(join(std::move(pp),2),d);
}
class Executor {
 std::shared_ptr<ttnn::MeshDevice> dev_;Json plan_;std::vector<char> blob_;bool wide_=false,trial_=false;
 DT activation_=DT::BFLOAT16,weight_dtype_=DT::BFLOAT16;ttnn::DeviceComputeKernelConfig compute_,resize_compute_;
 std::map<std::string,std::pair<T,T>> weights_;std::vector<std::optional<T>> staged_;
 size_t prepare_budget_=4294967296ULL,staging_budget_=4294967296ULL,staged_packed_bytes_=0;
 std::optional<T> input_,output_,rank32_;std::optional<ttnn::MeshTraceId> trace_;bool tuned_=false;
 T constant(const Json&j,bool bias=false,uint32_t start=0,uint32_t count=0){size_t n=size_t(j.at("bytes").as_int())/4,off=size_t(j.at("offset").as_int());if(off+n*4>blob_.size())throw std::runtime_error("weight range outside blob");std::vector<float> v(n);std::memcpy(v.data(),blob_.data()+off,n*4);auto d=j.at("shape").as_array();A s=bias?A{1,1,1,uint32_t(n)}:A{uint32_t(d[0].as_int()),uint32_t(d[1].as_int()),uint32_t(d[2].as_int()),uint32_t(d[3].as_int())};if(count){size_t block=bias?1:size_t(s[1])*s[2]*s[3];if((size_t(start)+count)*block>v.size())throw std::runtime_error("constant slice range");v=std::vector<float>(v.begin()+start*block,v.begin()+(start+count)*block);if(bias)s[3]=count;else s[0]=count;}return tensor(v,s,DT::FLOAT32);}
 F raw_conv(const Json& op,const F& x,const std::string& key,std::optional<std::array<uint32_t,4>> pads=std::nullopt,uint32_t channel0=0,uint32_t count=0){
  auto k=op.at("kernel").as_array(),st=op.at("stride").as_array(),di=op.at("dilation").as_array(),pd=op.at("padding").as_array();
  uint32_t cout=uint32_t(op.at("shape").as_array()[1].as_int()),groups=uint32_t(I(op,"groups"));
  std::array<uint32_t,2> kernel{uint32_t(k[0].as_int()),uint32_t(k[1].as_int())},stride{uint32_t(st[0].as_int()),uint32_t(st[1].as_int())},dilation{uint32_t(di[0].as_int()),uint32_t(di[1].as_int())};
  // SDK order is top,bottom,left,right; ONNX is top,left,bottom,right.
  std::array<uint32_t,4> pad=pads.value_or(std::array<uint32_t,4>{uint32_t(pd[0].as_int()),uint32_t(pd[2].as_int()),uint32_t(pd[1].as_int()),uint32_t(pd[3].as_int())});
  if(count){cout=count;groups=count;}
  if(!weights_.count(key)){
   T w=constant(op.at("weight"),false,channel0,count),b=constant(op.at("bias"),true,channel0,count);
   weights_.emplace(key,std::make_pair(w,b));
  }
  auto& wb=weights_.at(key);ttnn::operations::conv::Conv2dConfig cfg;cfg.weights_dtype=weight_dtype_;cfg.deallocate_activation=false;cfg.output_layout=ttnn::TILE_LAYOUT;cfg.config_tensors_in_dram=true;
  if(tuned_){const auto name=op.at("output").as_string();using ML=tt::tt_metal::TensorMemoryLayout;
   const bool stem=has(name,"/backbone/stem/stem.0/")||has(name,"/backbone/stem/stem.3/")||has(name,"/backbone/stem/stem.6/");
   const bool sep_dw=has(name,"/sep_bottleneck.0/depthwise_conv/")||has(name,"/sep_bottleneck.1/depthwise_conv/");
   if(sep_dw)cfg.shard_layout=ML::HEIGHT_SHARDED;                       // 'height'
   else{cfg.act_block_h_override=256;cfg.enable_act_double_buffer=true;cfg.enable_weights_double_buffer=true;  // 'abh256_dbuf'
    if(stem)cfg.shard_layout=ML::HEIGHT_SHARDED;}}                       // 'height_t'
  auto r=ttnn::operations::conv::conv2d::conv2d_DRAM(rows(x),wb.first,dev_.get(),x.c,cout,1,x.h,x.w,kernel,stride,pad,dilation,groups,activation_,wb.second,cfg,compute_,dram);
  wb.first=std::get<3>(r);if(std::get<4>(r))wb.second=*std::get<4>(r);
  return {std::get<0>(r),std::get<1>(r),std::get<2>(r),cout};
 }
 F conv(const Json&op,const F&x,const std::string&key,uint32_t channel0=0,uint32_t count=0){
  uint32_t cout=uint32_t(op.at("shape").as_array()[1].as_int()),groups=uint32_t(I(op,"groups"));auto k=op.at("kernel").as_array();auto di=op.at("dilation").as_array();auto st=op.at("stride").as_array();auto pd=op.at("padding").as_array();
  bool depth=groups>1&&groups==x.c&&groups==cout;
  if(tuned_){
   if(!count&&depth&&x.c>32&&di[0].as_int()>=24){const uint32_t width=has(op.at("output").as_string(),"/aspp_modules.2/depthwise_conv/")?128:32;
    auto src=rows(x);std::vector<T> parts;
    for(uint32_t c=0;c<x.c;c+=width){auto ce=std::min(c+width,x.c);F part{cut(src,{0,0,0,c},{1,x.h,x.w,ce}),x.h,x.w,ce-c};auto f=raw_conv(op,part,key+".C"+std::to_string(c),std::nullopt,c,ce-c);parts.push_back(rows(f));}
    auto t=join(std::move(parts),3);auto d=dims(t);return {tile(shape(t,{1,1,d[1]*d[2],d[3]})),d[1],d[2],x.c};}
   return raw_conv(op,x,key,std::nullopt,channel0,count);
  }
  if(!count&&depth&&x.c>32&&(wide_||di[0].as_int()>=24)){
   auto src=rows(x);std::vector<T> parts;
   for(uint32_t c=0;c<x.c;c+=32){auto ce=std::min(c+32,x.c);F part{cut(src,{0,0,0,c},{1,x.h,x.w,ce}),x.h,x.w,ce-c};auto f=conv(op,part,key+".C"+std::to_string(c),c,ce-c);parts.push_back(rows(f));}
   auto t=join(std::move(parts),3);auto d=dims(t);return {tile(shape(t,{1,1,d[1]*d[2],d[3]})),d[1],d[2],x.c};
  }
  bool point=k[0].as_int()==1&&k[1].as_int()==1&&st[0].as_int()==1&&st[1].as_int()==1&&pd[0].as_int()==0&&pd[1].as_int()==0&&groups==1;
  if((wide_||x.c%32!=0)&&point&&x.h*x.w>512){auto src=shape(rows(x),{1,1,x.h*x.w,x.c});std::vector<T> parts;
   for(uint32_t p=0;p<x.h*x.w;p+=512){auto pe=std::min(p+512,x.h*x.w);auto v=raw_conv(op,{cut(src,{0,0,p,0},{1,1,pe,x.c}),1,pe-p,x.c},key+".M"+std::to_string(pe-p));auto d=dims(v.t);parts.push_back(shape(v.t,{1,1,pe-p,d[3]}));}
   return {join(std::move(parts),2),x.h,x.w,cout};
  }
  if((wide_||(depth&&x.c%32!=0))&&!point&&(x.h>16||x.w>32)){
   uint32_t kh=uint32_t(k[0].as_int()),kw=uint32_t(k[1].as_int()),sh=uint32_t(st[0].as_int()),sw=uint32_t(st[1].as_int()),dh=uint32_t(di[0].as_int()),dw=uint32_t(di[1].as_int());
   int ph=int(pd[0].as_int()),pw=int(pd[1].as_int());uint32_t oh=(x.h+2*ph-dh*(kh-1)-1)/sh+1,ow=(x.w+2*pw-dw*(kw-1)-1)/sw+1;
   auto src=rows(x);std::vector<T> rr;
   for(uint32_t h=0;h<oh;h+=16){std::vector<T> cc;auto he=std::min(h+16,oh);
    for(uint32_t w=0;w<ow;w+=32){auto we=std::min(w+32,ow);int a=int(h*sh)-ph,b=int((he-1)*sh+dh*(kh-1)+1)-ph,c=int(w*sw)-pw,e=int((we-1)*sw+dw*(kw-1)+1)-pw;
     uint32_t loh=std::max(0,a),hih=std::min(int(x.h),b),low=std::max(0,c),hiw=std::min(int(x.w),e);std::array<uint32_t,4> pad{uint32_t(std::max(0,-a)),uint32_t(std::max(0,b-int(x.h))),uint32_t(std::max(0,-c)),uint32_t(std::max(0,e-int(x.w)))};
     std::string g=key+".G"+std::to_string(hih-loh)+"x"+std::to_string(hiw-low);for(auto q:pad)g+="_"+std::to_string(q);
     auto v=raw_conv(op,{cut(src,{0,loh,low,0},{1,hih,hiw,x.c}),hih-loh,hiw-low,x.c},g,pad,channel0,count);if(v.h!=he-h||v.w!=we-w)throw std::runtime_error("spatial tile coverage mismatch");auto d=dims(v.t);cc.push_back(shape(v.t,{1,v.h,v.w,d[3]}));}
    rr.push_back(join(std::move(cc),2));}
   return {join(std::move(rr),1),oh,ow,count?count:cout};
  }
  return raw_conv(op,x,key,std::nullopt,channel0,count);
 }
 F resize(const F&x,uint32_t th,uint32_t tw){auto src=rows(x);if(x.h==1&&x.w==1)return {ttnn::repeat(src,S({1,th,tw,1})),th,tw,x.c};auto original=src.dtype();if(wide_)src=cast_rows(src,DT::BFLOAT16);uint32_t c=dims(src)[3],padded=(c+31)/32*32;
  if(padded!=c)src=ttnn::pad(src,ttnn::SmallVector<std::array<uint32_t,2>>{{0,0},{0,0},{0,0},{0,padded-c}},0.0f,false,dram);
  if(x.h==1&&x.w==1){auto r=ttnn::repeat(src,S({1,th,tw,1}));if(wide_)r=cast_rows(r,original);return {r,th,tw,x.c};}
  uint32_t fh=th/x.h,fw=tw/x.w;std::vector<T> cp;
  for(uint32_t ch=0;ch<padded;ch+=(wide_?32:padded)){auto ce=std::min(ch+(wide_?32:padded),padded);std::vector<T> hp;
   for(uint32_t h=0;h<x.h;h+=8){auto he=std::min(h+8,x.h),lo=h?h-1:0,hi=std::min(he+1,x.h);auto p=cut(src,{0,lo,0,ch},{1,hi,x.w,ce});p=ttnn::upsample(p,tt::tt_metal::Array2D{fh,fw},"bilinear",dram,resize_compute_);hp.push_back(cut(p,{0,(h-lo)*fh,0,0},{1,(he-lo)*fh,tw,ce-ch}));}
   cp.push_back(join(std::move(hp),1));}
  auto out=join(std::move(cp),3);if(wide_)out=cast_rows(out,original);return {out,th,tw,x.c};
 }
 F pool(const F&x){auto src=rows(x);auto dtype=src.dtype();if(wide_)src=cast_rows(src,DT::BFLOAT16);uint32_t oh=(x.h+1)/2,ow=(x.w+1)/2;std::vector<T> hp;
  const uint32_t prow=tuned_?32:8;
  for(uint32_t h=0;h<oh;h+=prow){uint32_t he=std::min(h+prow,oh);int first=int(2*h)-1,last=int(2*he);uint32_t lo=std::max(0,first),hi=std::min(int(x.h),last);auto part=cut(src,{0,lo,0,0},{1,hi,x.w,x.c});
   auto y=ttnn::max_pool2d(part,1,hi-lo,x.w,x.c,tt::tt_metal::Array2D{3,3},tt::tt_metal::Array2D{2,2},tt::tt_metal::Array4D{uint32_t(std::max(0,-first)),uint32_t(std::max(0,last-int(x.h))),1,1},tt::tt_metal::Array2D{1,1},false,dram,std::nullopt,false);
   hp.push_back(shape(y.at(0),{1,he-h,ow,x.c}));}
  auto out=join(std::move(hp),1);if(wide_)out=cast_rows(out,dtype);return {out,oh,ow,x.c};
 }
 F gap(const F&x){uint32_t pixels=x.h*x.w;auto src=shape(tile(rows(x)),{1,1,pixels,x.c});
  if(!wide_)return {ttnn::mean(src,2,true,dram),1,1,x.c};std::vector<T> pp;
  for(uint32_t p=0;p<pixels;p+=512){uint32_t pe=std::min(p+512,pixels);auto t=ttnn::mean(tile(cut(src,{0,0,p,0},{1,1,pe,x.c})),2,true,dram);if(pixels>512)t=ttnn::multiply(t,float(pe-p)/float(pixels),std::nullopt,dram);pp.push_back(t);}
  while(pp.size()>1){std::vector<T> n;for(size_t i=0;i<pp.size();i+=2)n.push_back(i+1<pp.size()?ttnn::add(pp[i],pp[i+1],std::nullopt,dram):pp[i]);pp=std::move(n);}return {pp[0],1,1,x.c};
 }
 T argmax32(const T&crop,uint32_t h,uint32_t w){  // tile ArgMax over 32 channels (19 classes + pads at -10000)
  const std::optional<std::variant<int,ttnn::SmallVector<int>>> dim_h{std::in_place,-2};
  auto x=ttnn::to_layout(shape(ttnn::to_memory_config(crop,dram),{1,1,h*w,32}),ttnn::TILE_LAYOUT);
  x=ttnn::transpose(x,-2,-1,dram);                                   // (1,1,32,P)
  auto m=ttnn::max(x,dim_h,true,dram);                               // per-pixel max
  auto hit=ttnn::eq(x,m,std::nullopt,dram);                          // 1 where class == max
  auto best=ttnn::max(ttnn::multiply(hit,*rank32_,std::nullopt,dram),dim_h,true,dram);  // max(19-c): lowest index on ties
  auto idx=shape(ttnn::rsub(best,float(kClasses),std::nullopt,dram),{1,1,h,w});
  return ttnn::to_layout(idx,ttnn::ROW_MAJOR_LAYOUT);  // BF16 class index, as the validated Python graph
 }
 F tail(const F&x,uint32_t th,uint32_t tw){  // fused final bilinear resize + ArgMax, chunks planned for 20 cores x 2 rows
  auto src=rows(x);uint32_t c=dims(src)[3];if(c!=kClasses)throw std::runtime_error("tail expects 19 classes");
  src=ttnn::pad(src,ttnn::SmallVector<std::array<uint32_t,2>>{{0,0},{0,0},{0,0},{0,32-c}},-10000.0f,true,dram);
  const uint32_t fh=th/x.h,fw=tw/x.w;std::vector<T> maps;
  for(auto [s,e]:plan_chunks(x.h,2)){uint32_t lo=s?s-1:0,hi=std::min(e+1,x.h);
   auto p=ttnn::upsample(cut(src,{0,lo,0,0},{1,hi,x.w,32}),tt::tt_metal::Array2D{fh,fw},"bilinear",dram,resize_compute_);
   maps.push_back(argmax32(cut(p,{0,(s-lo)*fh,0,0},{1,(e-lo)*fh,tw,32}),(e-s)*fh,tw));}
  // uint32 for the executor output contract. Done once on the tile-aligned full map (1024 x 2048: no tile padding):
  // trial run_20261114T174302_110448Z showed typecast on the per-chunk reshaped TILE index maps (heights 108..156,
  // not multiples of 32) corrupts every tile row after the first. Values are integers 0..19, so the cast is exact.
  auto labels=ttnn::to_layout(join(std::move(maps),2),ttnn::TILE_LAYOUT);
  return {ttnn::to_layout(ttnn::typecast(labels,DT::UINT32,dram),ttnn::ROW_MAJOR_LAYOUT),th,tw,1};
 }
 void check_prepare_memory(){std::ifstream f("/proc/self/status");std::string line;while(std::getline(f,line))if(line.rfind("VmRSS:",0)==0){size_t kib=std::stoull(line.substr(6));if(kib>prepare_budget_/1024)throw std::runtime_error("NativePrepareMemoryBudgetExceeded; stop before next operator; no automatic retry/reset");return;}}
 T graph(){std::map<std::string,F> values;std::map<std::string,int> uses;
  for(const auto&o:plan_.at("ops").as_array())for(const auto&i:o.at("inputs").as_array())++uses[i.as_string()];
  values.emplace(plan_.at("input").at("name").as_string(),F{tuned_?shape(*input_,{1,1024,2048,3}):*input_,1024,2048,3});std::optional<T> final;
  for(const auto&op:plan_.at("ops").as_array()){
   check_prepare_memory();
   if(trial_)std::fprintf(stderr,"EAGLE_TRIAL OP %s %s\n",op.at("kind").as_string().c_str(),op.at("output").as_string().c_str());
   auto kind=op.at("kind").as_string(),out=op.at("output").as_string();const auto&ins=op.at("inputs").as_array();F x=values.at(ins[0].as_string());std::optional<F> y;
   if(kind=="Conv")y=conv(op,x,out);
   else if(kind=="Relu")y=F{ttnn::relu(x.t,dram),x.h,x.w,x.c};
   else if(kind=="Add"&&tuned_&&tile_ready(x)&&tile_ready(values.at(ins[1].as_string()))){auto b=values.at(ins[1].as_string());y=F{ttnn::add(ttnn::to_memory_config(x.t,dram),ttnn::to_memory_config(b.t,dram),std::nullopt,dram),x.h,x.w,x.c};}
   else if(kind=="Add"){auto b=values.at(ins[1].as_string());auto a=tile(shape(rows(x),{1,1,x.h*x.w,x.c}));auto bt=tile(shape(rows(b),{1,1,b.h*b.w,b.c}));y=F{ttnn::add(a,bt,std::nullopt,dram),x.h,x.w,x.c};}
   else if(kind=="MaxPool")y=pool(x);
   else if(kind=="GlobalAveragePool")y=gap(x);
   else if(kind=="Resize"&&tuned_&&x.c==kClasses){auto s=op.at("shape").as_array();y=tail(x,uint32_t(s[2].as_int()),uint32_t(s[3].as_int()));}
   else if(kind=="Resize"){auto s=op.at("shape").as_array();y=resize(x,uint32_t(s[2].as_int()),uint32_t(s[3].as_int()));}
   else if(kind=="Concat"){std::vector<T> parts;uint32_t c=0;for(const auto&i:ins){auto f=values.at(i.as_string());parts.push_back(rows(f));c+=f.c;}auto t=join(std::move(parts),3);y=F{t,x.h,x.w,c}; /* Keep NHWC row-major as validated Python concat; Conv prepares/shards it. */}
   else if(kind=="ArgMax"&&tuned_){if(x.c!=1)throw std::runtime_error("tuned ArgMax expects fused tail labels");y=x;}
   else if(kind=="ArgMax"){auto t=shape(rows(x),{1,1,x.h*x.w,x.c});auto a=ttnn::argmax(t,-1,true,std::nullopt,true,dram);y=F{a,x.h,x.w,1};}
   else if(kind=="Cast"){final=x.t;} // Final storage conversion is host-side, no giant device reshape.
   else throw std::runtime_error("unhandled data operator "+kind);
   if(y)values.emplace(out,*y);
   for(const auto&i:ins)if(--uses[i.as_string()]==0)values.erase(i.as_string());
  }
  if(!final)throw std::runtime_error("graph has no native ArgMax output");return *final;
 }
 A input_shape()const{return tuned_?A{1,1,1024,6144}:A{1,1024,2048,3};}
 public:
 explicit Executor(const Json&spec){const auto&p=spec.at("program");plan_=Json::parse(readf(p.at("plan_path").as_string()));auto blob=readf(p.at("weights_path").as_string());blob_.assign(blob.begin(),blob.end());const auto variant=plan_.at("variant").as_string();if(variant!="r18"&&variant!="r50"&&variant!="r101")throw std::runtime_error("unsupported DeepLab variant");wide_=variant=="r50";tuned_=variant=="r18";
  activation_=wide_?DT::FLOAT32:DT::BFLOAT16;weight_dtype_=activation_;auto opt=spec.at("backend").at("options");
  if(opt.has("staging_budget_bytes"))staging_budget_=size_t(opt.at("staging_budget_bytes").as_int());
  if(opt.has("prepare_host_budget_bytes"))prepare_budget_=size_t(opt.at("prepare_host_budget_bytes").as_int());check_prepare_memory();
  trial_=opt.has("trial_unqualified")&&opt.at("trial_unqualified").as_bool();
  if(trial_&&variant!="r18")throw std::runtime_error("Unqualified trial restricted to R18");
  std::fprintf(stderr,"EAGLE prepare: opening device 0\n");
  dev_=ttnn::open_mesh_device(0,32768,size_t(opt.at("trace_region_bytes").as_int()));dev_->enable_program_cache();
  compute_=ttnn::init_device_compute_kernel_config(dev_->arch(),std::nullopt,MathFidelity::HiFi4,false,true,false);
  resize_compute_=ttnn::init_device_compute_kernel_config(dev_->arch(),std::nullopt,MathFidelity::HiFi4,false,false,false);
  if(tuned_){std::vector<float> r(32,0.0f);for(uint32_t c=0;c<kClasses;++c)r[c]=float(kClasses-c);
   rank32_=ttnn::to_device(ttnn::to_layout(tensor(r,{1,1,32,1},DT::BFLOAT16),ttnn::TILE_LAYOUT),dev_.get(),dram);}
  // Tuned: one 12 KiB row-major page per image row (1024 pages) instead of one 6-byte page per pixel; the graph
  // re-pages it on the device. Same BF16 values.
  std::vector<float> zero(1024*2048*3,0.0f);input_=ttnn::to_device(tensor(zero,input_shape(),activation_),dev_.get(),dram);
  if(trial_){std::fprintf(stderr,"EAGLE_TRIAL ready; graph runs once at submit, no zero warm-up or trace\n");return;}
  {auto warm=graph();tt::tt_metal::distributed::Synchronize(dev_.get(),std::nullopt);}
  trace_=ttnn::operations::trace::begin_trace_capture(dev_.get(),ttnn::QueueId{0});output_=graph();ttnn::operations::trace::end_trace_capture(dev_.get(),*trace_,ttnn::QueueId{0});tt::tt_metal::distributed::Synchronize(dev_.get(),std::nullopt);
  if(output_->tensor_spec().compute_packed_buffer_size_bytes()!=2097152ULL*4)throw std::runtime_error("ArgMax output not packed contiguous uint32");
 }
 ~Executor() noexcept(false){if(dev_){if(trace_)ttnn::operations::trace::release_trace(dev_.get(),*trace_);staged_.clear();output_.reset();input_.reset();weights_.clear();dev_->close();}}
 void stage(int slot,const void*data,size_t n){if(slot<0||n!=25165824)throw std::runtime_error("stage shape/slot");const float*chw=static_cast<const float*>(data);std::vector<float> nhwc(3*1024*2048);for(size_t p=0;p<1024*2048;++p)for(size_t c=0;c<3;++c)nhwc[p*3+c]=chw[c*1024*2048+p];
  check_prepare_memory();auto prepared=tensor(nhwc,input_shape(),activation_);size_t bytes=prepared.tensor_spec().compute_packed_buffer_size_bytes();
  size_t old=size_t(slot)<staged_.size()&&staged_[slot]?staged_[slot]->tensor_spec().compute_packed_buffer_size_bytes():0;
  if(bytes>staging_budget_||staged_packed_bytes_-old>staging_budget_-bytes)throw std::runtime_error("StagingPackedTensorBudgetExceeded; physical padding included");
  if(size_t(slot)>=staged_.size())staged_.resize(size_t(slot)+1);staged_[size_t(slot)]=std::move(prepared);staged_packed_bytes_=staged_packed_bytes_-old+bytes;check_prepare_memory();
 }
 void run(int slot,uint32_t*out,size_t pixels){if(pixels!=2097152||slot<0||size_t(slot)>=staged_.size()||!staged_[slot])throw std::runtime_error("unstaged request");
  tt::tt_metal::write_tensor(*staged_[slot],*input_,false,ttnn::QueueId{0});
  if(trial_){output_=graph();tt::tt_metal::distributed::Synchronize(dev_.get(),std::nullopt);auto logical=output_->to_vector<uint32_t>();if(logical.size()!=pixels)throw std::runtime_error("Logical ArgMax output shape differs");std::copy(logical.begin(),logical.end(),out);return;}
  else ttnn::operations::trace::execute_trace(dev_.get(),*trace_,ttnn::QueueId{0},true);
  if(output_->tensor_spec().compute_packed_buffer_size_bytes()!=pixels*sizeof(uint32_t))throw std::runtime_error("Output packing differs from caller buffer; refuse D2H");
  tt::tt_metal::memcpy(out,*output_,std::nullopt,true);
 }
};
inline void err(char*out,size_t cap,const char*m){if(cap)std::snprintf(out,cap,"%s",m);}
void* prepare(const char*config,char*e,size_t cap){try{return new Executor(Json::parse(config));}catch(const std::exception&x){err(e,cap,x.what());return nullptr;}}
bool stage(void*p,int slot,const void*d,size_t n,char*e,size_t cap){try{static_cast<Executor*>(p)->stage(slot,d,n);return true;}catch(const std::exception&x){err(e,cap,x.what());return false;}}
bool run(void*p,int slot,uint32_t*out,size_t n,char*e,size_t cap){try{static_cast<Executor*>(p)->run(slot,out,n);return true;}catch(const std::exception&x){err(e,cap,x.what());return false;}}
bool destroy(void*p,char*e,size_t cap){try{delete static_cast<Executor*>(p);return true;}catch(const std::exception&x){err(e,cap,x.what());return false;}catch(...){err(e,cap,"unknown cleanup failure");return false;}}
const char* describe(){return R"({"eagle_n.runtime":"BOS TTNN native C++","eagle_n.silicon_revision":"not_independently_observed","eagle_n.firmware":"not_observable","eagle_n.device_id":0,"eagle_n.profiler":"off","eagle_n.completion":"synchronous trace replay then blocking D2H","eagle_n.device_timing":null,"eagle_n.graph":"r18: tuned 2026-11-15 (fused tile ArgMax, per-layer conv config); r50/r101: base"})";}
} // namespace sdv::eagle
extern "C" const sdv::eagle::ExecutorAPI* sdv_eagle_executor_v1(){static const sdv::eagle::ExecutorAPI a{sdv::eagle::kExecutorABI,true,sdv::eagle::describe,sdv::eagle::prepare,sdv::eagle::stage,sdv::eagle::run,sdv::eagle::destroy};return &a;}

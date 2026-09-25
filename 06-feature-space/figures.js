// Figures for 06. Python supplies every computed coordinate and distance.
const NS = 'http://www.w3.org/2000/svg';
const C = {ink:'#1f2a37', muted:'#6b7785', grid:'#e6eaef', axis:'#b5bec8', blue:'#2a78d6', orange:'#eb6834', pale:'#f3f6f9'};
const DIGIT = ['#2457a6','#e07027','#277b59','#b53f56','#7656ad','#987024','#157f91','#ba5799','#687887','#af4a27'];
const root = document.getElementById('root');
const tip = document.getElementById('tip');
let width = 760, painter;
const sv = (tag, a={}, parent) => { const el=document.createElementNS(NS,tag); for(const [k,v] of Object.entries(a)) el.setAttribute(k,v); if(parent) parent.append(el); return el; };
const el = (tag, a={}, children=[]) => { const n=document.createElement(tag); for(const [k,v] of Object.entries(a)) k==='text'?n.textContent=v:k==='html'?n.innerHTML=v:n.setAttribute(k,v); children.forEach(c=>n.append(c)); return n; };
const label = (s,x,y,value,a={}) => { const n=sv('text',{x,y,...a},s); n.textContent=value; return n; };
const line = (s,x1,y1,x2,y2,color=C.axis,w=1,dash) => sv('line',{x1,y1,x2,y2,stroke:color,'stroke-width':w,...(dash?{'stroke-dasharray':dash}:{})},s);
const circle = (s,x,y,r,fill,stroke='#fff') => sv('circle',{cx:x,cy:y,r,fill,stroke,'stroke-width':1.2},s);
const ring = (s,x,y,r,color) => sv('circle',{cx:x,cy:y,r,fill:'#fff',stroke:color,'stroke-width':2},s);
function arrow(s,x1,y1,x2,y2,color=C.muted){line(s,x1,y1,x2,y2,color,2);const a=Math.atan2(y2-y1,x2-x1),h=7;
  sv('path',{d:`M${x2},${y2} L${x2-h*Math.cos(a-.45)},${y2-h*Math.sin(a-.45)} L${x2-h*Math.cos(a+.45)},${y2-h*Math.sin(a+.45)}Z`,fill:color},s);}
// Stack dots on a number line so that none overlap. Returns the row of each value.
function stackRows(xs,gap){const rows=[],out=new Array(xs.length);xs.map((x,i)=>[x,i]).sort((a,b)=>a[0]-b[0]).forEach(([x,i])=>{let r=0;while(rows[r]!==undefined&&x-rows[r]<gap)r++;rows[r]=x;out[i]=r;});return out;}
const scale = (a,b,c,d) => x => c+(x-a)/(b-a)*(d-c);
const range = values => { const lo=Math.min(...values),hi=Math.max(...values),p=(hi-lo)*.09||1;return [lo-p,hi+p]; };
const svgBox = (w,h,aria) => sv('svg',{width:'100%',viewBox:`0 0 ${w} ${h}`,role:'img','aria-label':aria,style:'margin:0 auto;max-width:'+w+'px'},root);
function imageCanvas(hex, pixels=100){
  const c=el('canvas',{width:28,height:28,role:'img','aria-label':'手書き数字の元画像',style:`width:${pixels}px;height:${pixels}px;image-rendering:pixelated;border:1px solid #d6dce3;background:#000`});
  const ctx=c.getContext('2d'),im=ctx.createImageData(28,28);
  for(let i=0;i<392;i++){ const b=parseInt(hex.slice(i*2,i*2+2),16);for(let j=0;j<2;j++){const v=((j?b&15:b>>4)*17),p=(i*2+j)*4;im.data[p]=v;im.data[p+1]=v;im.data[p+2]=v;im.data[p+3]=255;}}
  ctx.putImageData(im,0,0);return c;
}
function updatePreview(box,hex,answer,pixels=82){
  box.replaceChildren(imageCanvas(hex,pixels),el('div',{style:'font-size:14px;line-height:1.8',html:`<b>元画像</b><br>正解：<b>${answer}</b>`}));
}
function imagePreview(hex,answer,pixels=82){
  const box=el('div',{style:'display:flex;align-items:center;gap:16px;padding:10px 14px;border:1px solid #dce3e9;border-radius:9px;margin-top:10px','aria-live':'polite'});
  updatePreview(box,hex,answer,pixels);return box;
}
function legendDigits(){ const box=el('div',{class:'legend',style:'gap:2px 16px;margin:2px 0 8px'});DIGIT.forEach((color,i)=>box.append(el('span',{class:'key'},[el('span',{style:`display:inline-block;width:10px;height:10px;border-radius:50%;background:${color}`}),el('span',{text:String(i)})])));root.append(box); }
function seg(names,active,change,aria){const wrap=el('div',{class:'controls'}), group=el('div',{class:'seg',role:'group','aria-label':aria}); names.forEach((name,i)=>{const b=el('button',{type:'button','aria-pressed':String(i===active),text:name});b.onclick=()=>change(i);group.append(b);});wrap.append(group);root.append(wrap);}
function title(t){root.append(el('div',{style:'font-size:14px;font-weight:600;color:#1f2a37;margin:0 0 8px',text:t}));}
function caption(){if(CONFIG.caption)root.append(el('figcaption',{text:CONFIG.caption}));}
function axes(s,W,H,xmin,xmax,ymin,ymax,xname,yname,mobile=false){
  const m={l:mobile?34:48,r:14,t:22,b:mobile?36:43};
  const x=scale(xmin,xmax,m.l,W-m.r),y=scale(ymin,ymax,H-m.b,m.t);
  for(let i=0;i<5;i++){const xx=m.l+(W-m.l-m.r)*i/4, yy=m.t+(H-m.t-m.b)*i/4;line(s,m.l,yy,W-m.r,yy,C.grid);line(s,xx,H-m.b,xx,H-m.b+4);}
  line(s,m.l,H-m.b,W-m.r,H-m.b,C.axis,1.5);line(s,m.l,m.t,m.l,H-m.b,C.axis,1.5);
  label(s,W-m.r,H-5,xname,{'text-anchor':'end',class:'ink'});label(s,m.l,14,yname,{class:'ink'});
  return {x,y,m};
}
function placeTip(e){const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-5);const y=e.clientY+14+tip.offsetHeight<innerHeight?e.clientY+14:e.clientY-tip.offsetHeight-14;tip.style.transform=`translate(${Math.max(5,x)}px,${Math.max(5,y)}px)`;}
function tooltip(e,hex,answer){tip.replaceChildren(imageCanvas(hex,76),el('span',{html:`元画像<br><b>正解：${answer}</b>`}));tip.style.opacity=1;placeTip(e);}
function hideTip(){tip.style.opacity=0;}
function plotMap(points,labels,opts={}){
  const compact=width<560,W=compact?360:760,H=compact?350:460;
  const s=svgBox(W,H,opts.title||'数字画像の2次元配置');
  const [xmin,xmax]=range(points.map(p=>p[0])),[ymin,ymax]=range(points.map(p=>p[1]));
  const a=axes(s,W,H,xmin,xmax,ymin,ymax,opts.xname||'UMAP 1',opts.yname||'UMAP 2',compact);
  const order=points.map((_,i)=>i);if(opts.selected!==undefined)order.sort((u,v)=>(u===opts.selected)-(v===opts.selected));
  order.forEach(i=>{const [px,py]=points[i],x=a.x(px),y=a.y(py),chosen=i===opts.selected;
    const dot=circle(s,x,y,chosen?7.5:compact?2.8:3.6,DIGIT[labels[i]],chosen?C.ink:'#fff');dot.setAttribute('style','cursor:pointer');
    dot.addEventListener('click',()=>opts.onSelect?.(i));
    if(opts.images){dot.addEventListener('pointerenter',e=>tooltip(e,opts.images[i],labels[i]));dot.addEventListener('pointermove',placeTip);dot.addEventListener('pointerleave',hideTip);}
  });
}
const FIG={};
FIG.pipeline=d=>{const compact=width<560,W=compact?360:760,H=compact?275:150,s=svgBox(W,H,'MNIST画像が128個の特徴量と10個のスコアへ変わる');
  const items=compact?[[18,5,324,72,'784個の画素値'],[18,101,324,72,'128個の特徴量'],[18,197,324,72,'10個のスコア']]:[[10,25,210,105,'784個の画素値'],[278,25,210,105,'128個の特徴量'],[546,25,205,105,'10個のスコア']];
  items.forEach(([x,y,w,h,t],i)=>{sv('rect',{x,y,width:w,height:h,rx:9,fill:i===1?'#edf5fd':C.pale,stroke:i===1?C.blue:'#d9e0e7'},s);label(s,x+15,y+25,t,{class:'strong'});
    if(i===0){label(s,x+15,y+51,`28 × 28 の明るさ`,{class:'ink'});const size=compact?57:70;sv('image',{href:imageCanvas(d.image).toDataURL(),x:x+w-size-10,y:y+(h-size)/2,width:size,height:size,style:'image-rendering:pixelated'},s);}
    if(i===1) label(s,x+15,y+51,d.hidden.slice(0,5).map(n=>n.toFixed(1)).join(', ')+' …',{class:'ink',style:'font-size:12px'});
    if(i===2) label(s,x+15,y+51,`最大のスコア → ${d.scores.indexOf(Math.max(...d.scores))}`,{class:'ink'});
    if(i<2){const name=i===0?'隠れ層':'出力層';
      if(compact){arrow(s,110,y+h+4,110,y+h+20);label(s,122,y+h+17,name,{class:'ink',style:'font-size:12px'});}
      else{arrow(s,x+w+6,77,x+w+52,77);label(s,x+w+29,66,name,{'text-anchor':'middle',class:'ink',style:'font-size:12px'});}}
  });
};
FIG.neighbors=d=>{const wrap=el('div',{class:'row',style:'justify-content:center;gap:14px'});root.append(wrap);
  ['基準の画像','画素値で近い','特徴量で近い'].forEach((name,i)=>{const p=el('div',{style:'flex:1 1 165px;max-width:220px;text-align:center;padding:12px;border:1px solid #dce3e9;border-radius:9px'});p.append(el('div',{style:'font-size:13px;font-weight:600;margin-bottom:8px',text:name}),imageCanvas(d.images[i],84),el('div',{style:'font-size:13px;color:#4b5866;margin-top:5px',text:`正解：${d.labels[i]}`}));wrap.append(p);});
};
FIG.pcaDemo=d=>{let selected=0;return function draw(){seg(['横方向','縦方向','PCAの方向'],selected,i=>{selected=i;repaint();},'写す軸の向き');
  const view=d.views[selected];
  root.append(el('div',{class:'legend'},[el('span',{html:`<span style="color:${C.blue}">●</span> 元の点`}),el('span',{html:`<span style="color:${C.orange}">━</span> 写す軸（${view.angle.toFixed(0)}°）`}),
    el('span',{html:`<span style="color:${C.orange}">○</span> 軸に写した位置`}),el('span',{html:`残るばらつき <b>${Math.round(view.share*100)}%</b>`})]));
  // Left: equal scale on both axes so that the drawn angle is the real angle.
  const compact=width<560,R=3.2,P=compact?300:240,W=compact?360:760,left=compact?{w:348,h:358}:{w:302,h:305},H=compact?505:305;
  const s=svgBox(W,H,'点群と、写す軸、軸に写した位置の対応');
  const ax=axes(s,left.w,left.h,-R,R,-R,R,'特徴量1','特徴量2',compact);
  const theta=view.angle*Math.PI/180,v=[Math.cos(theta),Math.sin(theta)];
  line(s,ax.x(-3*v[0]),ax.y(-3*v[1]),ax.x(3*v[0]),ax.y(3*v[1]),C.orange,2.5);
  d.points.forEach((p,i)=>{const t=view.values[i];line(s,ax.x(p[0]),ax.y(p[1]),ax.x(t*v[0]),ax.y(t*v[1]),'#aab5c2',1,'3 3');});
  d.points.forEach((p,i)=>{const t=view.values[i];ring(s,ax.x(t*v[0]),ax.y(t*v[1]),3.2,C.orange);});
  d.points.forEach(p=>circle(s,ax.x(p[0]),ax.y(p[1]),4.8,C.blue));
  // Right (below on phones): the same positions laid flat. The scale is fixed across views.
  const x0=compact?34:380,x1=compact?346:740,ybase=compact?480:205,xp=scale(-R,R,x0,x1);
  label(s,x0,ybase-(compact?88:80),'軸に写した位置（どの向きでも同じ目盛り）',{class:'strong'});
  line(s,x0,ybase,x1,ybase,C.orange,2.5);
  const rows=stackRows(view.values.map(xp),10);
  view.values.forEach((t,i)=>ring(s,xp(t),ybase-9-rows[i]*10,4.2,C.orange));
  label(s,x0,ybase+20,'−',{});label(s,x1,ybase+20,'＋',{'text-anchor':'end'});
  };};
FIG.scatter=d=>{let selected;return function draw(){title(d.title);legendDigits();
  plotMap(d.points,d.labels,{selected,xname:d.xname,yname:d.yname,images:d.images,onSelect:i=>{selected=i;repaint();}});
  if(selected!==undefined)root.append(imagePreview(d.images[selected],d.labels[selected]));
  };};
FIG.isomap=d=>{let selected=0;return function draw(){seg(d.views.map(v=>`近傍 ${v.k}点`),selected,i=>{selected=i;repaint();},'各点を結ぶ近傍の数');
  const v=d.views[selected],n=d.points.length,upper=i=>i<n/2;
  root.append(el('div',{class:'legend'},[el('span',{html:`<span style="color:${C.blue}">●</span> 紙の上側`}),el('span',{html:`<span style="color:${C.orange}">○</span> 紙の下側`}),
    el('span',{html:'<span style="color:#aab5c2">―</span> 近傍の接続'}),el('span',{html:`<span style="color:${C.ink}">━</span> 最短経路`}),
    el('span',{html:`始点→終点　経路 <b>${v.distance.toFixed(2)}</b> ／ 直線 <b>${d.direct.toFixed(2)}</b>`})]));
  const compact=width<560,W=compact?360:760,H=compact?290:300,s=svgBox(W,H,'折り返した紙の近傍グラフと、1本の軸に並べた結果');
  const dot=(i,x,y,big)=>upper(i)?circle(s,x,y,big?6:3.6,C.blue):ring(s,x,y,big?5.5:3.2,C.orange);
  label(s,compact?4:10,16,'① 近くの点を結び、② 最短経路をたどる',{class:'strong'});
  const x=scale(-4.5,4.8,compact?28:45,W-25),y=scale(-1,1,compact?150:160,compact?30:34);
  v.edges.forEach(([i,j])=>line(s,x(d.points[i][0]),y(d.points[i][1]),x(d.points[j][0]),y(d.points[j][1]),'#c6d0da',1.2));
  v.path.slice(1).forEach((j,k)=>{const i=v.path[k];line(s,x(d.points[i][0]),y(d.points[i][1]),x(d.points[j][0]),y(d.points[j][1]),C.ink,3);});
  d.points.forEach((p,i)=>dot(i,x(p[0]),y(p[1]),i===0||i===n-1));
  label(s,x(d.points[0][0])-8,y(d.points[0][1])+4,'始点',{class:'strong','text-anchor':'end'});label(s,x(d.points.at(-1)[0])-8,y(d.points.at(-1)[1])+4,'終点',{class:'strong','text-anchor':'end'});
  // ③ Positions on one axis that keep the path distances as well as possible (fixed scale for both views).
  const top=compact?190:200,base=top+55,xl=scale(-9.6,9.6,compact?20:45,W-(compact?20:25));
  label(s,compact?4:10,top,'③ 経路の距離を保つように、1本の軸へ並べる',{class:'strong'});
  line(s,xl(-9.6),base,xl(9.6),base,C.axis,1.5);
  v.layout.forEach((t,i)=>dot(i,xl(t),upper(i)?base-9:base+9,i===0||i===n-1));
  label(s,xl(v.layout[0]),base-20,'始点',{class:'strong','text-anchor':'middle'});label(s,xl(v.layout[n-1]),base+32,'終点',{class:'strong','text-anchor':'middle'});
  };};
FIG.umapSteps=()=>{const compact=width<560,W=compact?360:760,H=compact?570:210,s=svgBox(W,H,'UMAPが近傍を結び、点の配置を更新する3段階');
  const names=['① 元の空間で近傍を探す','② つながりに強さを付ける','③ 2次元の点を動かす'];
  names.forEach((name,i)=>{const x=compact?15:8+i*254,y=compact?i*188+8:10,w=compact?330:238,h=166;
    sv('rect',{x,y,width:w,height:h,rx:9,fill:i===2?'#f0f6fc':C.pale,stroke:'#dce3e9'},s);label(s,x+12,y+25,name,{class:'strong',style:'font-size:13px'});
    const ps=[[45,88],[92,70],[73,117],[160,83],[184,128]].map(([px,py])=>[x+px*(w/238),y+py]);
    if(i<2){[[0,1],[0,2],[1,2],[3,4]].forEach(([a,b],j)=>line(s,...ps[a],...ps[b],C.muted,i===1?[4.5,2.5,1,3.5][j]:1.5));
      ps.forEach(([px,py],j)=>circle(s,px,py,5,j<3?C.blue:C.orange));}
    else{const start=[[48,100],[98,62],[130,112],[152,72],[187,124]].map(([px,py])=>[x+px*(w/238),y+py]);
      const end=[[65,92],[78,74],[77,112],[166,89],[181,112]].map(([px,py])=>[x+px*(w/238),y+py]);
      start.forEach(([px,py],j)=>{circle(s,px,py,4,'#fff','#aab5c2');line(s,px,py,end[j][0],end[j][1],C.muted,1.4,'3 2');});
      end.forEach(([px,py],j)=>circle(s,px,py,5,j<3?C.blue:C.orange));
      label(s,x+12,y+155,'○ 最初　● 更新後',{class:'ink',style:'font-size:12px'});}
  });};
FIG.compare=d=>{let view=0,selected=0;const names=['画素値 784','学習前 128','学習後 128'];return function draw(){seg(names,view,i=>{view=i;repaint();},'比較する表現');
  const key=['pixels','before','after'][view];
  title(names[view]+' → UMAPで2次元へ');
  root.append(el('div',{class:'legend'},[el('span',{html:`元の${view?128:784}次元で最も近い10枚のうち、同じ数字の割合 <b>${Math.round(d.agreement[key]*100)}%</b>`})]));
  legendDigits();plotMap(d.maps[['pixels','before','after'][view]],d.labels,{selected,images:d.images,onSelect:i=>{selected=i;repaint();}});
  root.append(imagePreview(d.images[selected],d.labels[selected]));
  };};
FIG.settings=d=>{let selected;return function draw(){legendDigits();const row=el('div',{class:'row'});root.append(row);
  for(const [u,points] of [[d.settings[0],d.left],[d.settings[1],d.right]]){const name=`近傍数 ${u.n_neighbors}・min_dist ${u.min_dist}`;
    const holder=el('div',{class:'panel'});row.append(holder);const count=root.childElementCount;
    title(name);plotMap(points,d.labels,{selected,images:d.images,onSelect:i=>{selected=i;repaint();}});
    while(root.childElementCount>count)holder.append(root.children[count]);
  }
  if(selected!==undefined)root.append(imagePreview(d.images[selected],d.labels[selected]));
  };};
function repaint(){hideTip();root.replaceChildren();const f=typeof painter==='function'?painter:()=>{};f();caption();requestAnimationFrame(fit);}
function fit(){if(window.frameElement)window.frameElement.style.height=Math.ceil(document.body.getBoundingClientRect().height)+'px';}
for(const key of ['pipeline','neighbors','umapSteps']){const draw=FIG[key];FIG[key]=data=>()=>draw(data);}
painter=FIG[CONFIG.name](CONFIG.data);
new ResizeObserver(()=>{const w=root.clientWidth;if(w&&w!==width){width=w;hideTip();repaint();}else fit();}).observe(document.body);
repaint();

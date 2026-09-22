// A frame is one complete parameter state, not a substep of its calculation.
const el = id => document.getElementById(id);
let pos = 0, playing = false, timer = null, generation = 0;

function drawPlot() {
  const width = Math.max(300, document.documentElement.clientWidth - 16);
  const height = 250, left = 49, right = width - 17, top = 30, bottom = 214;
  const sx = w => left + (w + 1.15) / 2.3 * (right - left);
  const sy = loss => bottom - (loss - .55) / 1.2 * (bottom - top);
  const line = points => points.map(([x,y])=>`${sx(x).toFixed(2)},${sy(y).toFixed(2)}`).join(' ');
  const state = DATA.states[pos], w = state.theta[0];
  const tangent = [[w-.18,state.loss-.18*state.gradient],[w+.18,state.loss+.18*state.gradient]];
  const trace = DATA.states.slice(0,pos+1).map(s=>[s.theta[0],s.loss]);
  const grid = [.75,1,1.5].map(y=>`<path d="M${left},${sy(y)}H${right}" stroke="#edf0f3"/><text x="${left-8}" y="${sy(y)+5}" text-anchor="end">${y}</text>`).join('');
  const ticks = [-1,0,1].map(x=>`<text x="${sx(x)}" y="235" text-anchor="middle">${x}</text>`).join('');
  // Keep the moving dot mounted so the short transition remains visible.
  if (!el('current')) {
    el('plot').innerHTML='<g id="drawing"></g><circle id="current" r="6" fill="#2563a6" stroke="white" stroke-width="2"/>';
  }
  el('plot').setAttribute('viewBox',`0 0 ${width} ${height}`);
  el('drawing').innerHTML=`${grid}<path d="M${left},${top}V${bottom}H${right}" fill="none" stroke="#a7b4bf"/>${ticks}<text x="${left}" y="17">損失 L</text><text x="${right}" y="248" text-anchor="end">重み w</text><polyline points="${line(DATA.curve)}" fill="none" stroke="#a6b7c3" stroke-width="2.5"/><polyline points="${line(trace)}" fill="none" stroke="#2563a6" stroke-width="3"/><polyline points="${line(tangent)}" fill="none" stroke="#bb6025" stroke-width="2"/>`;
  el('current').setAttribute('cx',sx(w));
  el('current').setAttribute('cy',sy(state.loss));
}

function render() {
  drawPlot();
  const state = DATA.states[pos];
  el('readout').textContent=`更新 ${pos} / ${DATA.states.length-1}　 w = ${state.theta[0].toFixed(3)}　 L = ${state.loss.toFixed(3)}`;
  el('back').disabled=pos===0;
  el('next').disabled=pos===DATA.states.length-1;
  el('play').textContent=playing?'一時停止':'ループ再生';
  el('play').setAttribute('aria-pressed',String(playing));
}

function stop() {
  playing=false;
  generation++;
  if(timer!==null)clearTimeout(timer);
  timer=null;
}

function schedule() {
  const token=generation;
  timer=setTimeout(()=>{
    if(!playing || token!==generation)return;
    timer=null;
    pos=(pos+1)%DATA.states.length;
    render();
    schedule();
  },1600);
}

function move(delta) {
  stop();
  pos=Math.max(0,Math.min(DATA.states.length-1,pos+delta));
  render();
}

el('back').onclick=()=>move(-1);
el('next').onclick=()=>move(1);
el('play').onclick=()=>{
  if(playing)stop();
  else {playing=true;generation++;schedule();}
  render();
};
document.addEventListener('visibilitychange',()=>{
  if(document.hidden){stop();render();}
});
window.addEventListener('pagehide',stop);
new ResizeObserver(drawPlot).observe(document.documentElement);
render();

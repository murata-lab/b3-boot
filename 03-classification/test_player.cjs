// The notebook supplies the actual data used by the lesson; no second Python module.
const {test}=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm');
const fs=require('node:fs'),path=require('node:path'),{execFileSync}=require('node:child_process');
const root=__dirname,python=process.env.PYTHON||path.join(root,'..','.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
const fixture=JSON.parse(execFileSync(python,['-c',`
import json, numpy as np, notebook, xml.etree.ElementTree as ET
_, d = notebook.classification_core.run(np=np)
X,Y,fg=d['X'],d['Y'],d['loss_gradient']
checks=[]
for theta in [np.array([-1.,0.]), np.array([.3,-.2]), np.array([2.,1.])]:
    analytic=fg(X,Y,theta)[1]
    numeric=np.array([(fg(X,Y,theta+np.eye(2)[j]*1e-6)[0]-fg(X,Y,theta-np.eye(2)[j]*1e-6)[0])/2e-6 for j in range(2)])
    checks.append(float(np.max(np.abs(analytic-numeric))))
for size in [6,2,1]:
    states=d['history'](epochs=6,batch_size=size,fixed_bias=False)
    rng=np.random.default_rng(31)
    theta=np.array([-1.,0.]);i=0
    for epoch in range(6):
        for ids in np.array_split(rng.permutation(6),6//size):
            theta=theta-.3*fg(X[ids],Y[ids],theta)[1];i+=1
            assert np.allclose(states[i]['theta'],theta)
            assert np.isclose(states[i]['loss'],fg(X,Y,theta)[0])
    assert len(states)==1+36//size
    assert all(-1.2<=s['theta'][0]<=1.2 and -.5<=s['theta'][1]<=.5 for s in states)
for name in ['sigmoid_figure','boundary_figure','loss_figure','rate_figure','method_figure','softmax_figure']:
    graphic=d[name]()
    ET.fromstring(graphic)
    assert not any(word in graphic for word in ['NaN','Infinity','undefined'])
p,loss=d['softmax_loss']([1000,999,998],2)
print(json.dumps(dict(data=d['demo_data'](), gradient_errors=checks, softmax=p.tolist(), softmax_loss=loss,
    rates={str(r):[s['loss'] for s in d['history'](eta=r)] for r in [.1,1,3]})))
`],{cwd:root,maxBuffer:1024*1024}).toString());
const source=fs.readFileSync(path.join(root,'player.js'),'utf8');
function setup(){
  const elements=new Map();let pending=[];
  function make(id){
    const e={id,textContent:'',disabled:false,attributes:{},setAttribute(k,v){this.attributes[k]=String(v);}};
    let content='';Object.defineProperty(e,'innerHTML',{get:()=>content,set:v=>{content=v;for(const m of v.matchAll(/id="([^"]+)"/g))if(!elements.has(m[1]))elements.set(m[1],make(m[1]));}});
    return e;
  }
  for(const id of ['plot','back','next','play','readout'])elements.set(id,make(id));
  const doc={documentElement:{clientWidth:760},getElementById:id=>elements.get(id)||null,addEventListener(){}};
  const context=vm.createContext({DATA:fixture.data,document:doc,window:{addEventListener(){}},ResizeObserver:class{observe(){}},setTimeout:fn=>{pending.push(fn);return pending.length;},clearTimeout(){}});
  vm.runInContext(source,context);
  return {e:elements,run:s=>vm.runInContext(s,context),tick:()=>{const q=pending;pending=[];q.forEach(fn=>fn());},click:id=>elements.get(id).onclick()};
}

test('the displayed gradient is the derivative of the actual mean loss',()=>{
  assert.ok(fixture.gradient_errors.every(e=>e<1e-8));
  const states=fixture.data.states;
  assert.equal(states.length,10);
  assert.ok(Math.abs(states[0].loss-1.496259016)<1e-6);
  for(let i=1;i<states.length;i++){
    assert.ok(Math.abs(states[i].theta[0]-(states[i-1].theta[0]-.3*states[i-1].gradient))<1e-12);
    assert.equal(states[i].theta[1],0);
    assert.ok(states[i].loss<states[i-1].loss);
  }
});
test('static comparisons and softmax match the explanation',()=>{
  assert.ok(fixture.rates['1'][9]<fixture.rates['0.1'][9]);
  assert.ok(fixture.rates['3'][1]>fixture.rates['3'][0]);
  assert.ok(fixture.rates['3'][9]>fixture.rates['1'][9]);
  assert.ok(Math.abs(fixture.softmax.reduce((a,b)=>a+b)-1)<1e-12);
  assert.ok(Math.abs(fixture.softmax_loss-2.407605964)<1e-9);
});
test('each forward action moves the dot one complete update, and back restores it',()=>{
  const h=setup();assert.equal(h.e.get('back').disabled,true);
  const initial=h.e.get('drawing').innerHTML;
  let previous=h.e.get('current').attributes.cx;
  for(let i=1;i<=9;i++){
    h.click('next');assert.equal(h.run('pos'),i);
    assert.notEqual(h.e.get('current').attributes.cx,previous);
    previous=h.e.get('current').attributes.cx;
    assert.match(h.e.get('readout').textContent,new RegExp(`更新 ${i} / 9`));
    assert.ok(!/NaN|Infinity|undefined/.test(h.e.get('drawing').innerHTML));
  }
  assert.equal(h.e.get('next').disabled,true);h.click('next');assert.equal(h.run('pos'),9);
  for(let i=0;i<9;i++)h.click('back');
  assert.equal(h.e.get('drawing').innerHTML,initial);h.click('back');assert.equal(h.run('pos'),0);
});
test('automatic playback loops from the last state to the initial state',()=>{
  const h=setup();h.click('play');
  for(let i=1;i<=10;i++){h.tick();assert.equal(h.run('pos'),i%10);}
  assert.equal(h.run('playing'),true);
  assert.equal(h.e.get('play').attributes['aria-pressed'],'true');
});
test('pause and manual movement invalidate old timer callbacks',()=>{
  const h=setup();h.click('play');h.click('play');h.tick();assert.equal(h.run('pos'),0);
  h.click('play');h.click('next');h.tick();assert.equal(h.run('pos'),1);
  assert.equal(h.run('playing'),false);
  h.click('play');h.click('play');h.click('play');h.tick();assert.equal(h.run('pos'),2);
});
test('the lesson has one player, three controls, and no optional interaction panels',()=>{
  const html=fs.readFileSync(path.join(root,'player.html'),'utf8');
  const notebook=fs.readFileSync(path.join(root,'notebook.py'),'utf8');
  assert.equal((html.match(/<button\b/g)||[]).length,3);
  assert.equal((notebook.match(/mo\.iframe\(/g)||[]).length,1);
  assert.ok(!/<(?:input|select|details)\b/.test(html));
  assert.ok(!/mo\.(?:ui|accordion)|任意|360コマ|計算の中身を見る/.test(notebook));
  assert.deepEqual(fs.readdirSync(root).filter(f=>f.endsWith('.py')),['notebook.py']);
});

/* A deliberately schematic storyboard. The curve is not a simulation result. */
const MIN_BUDGET=4,MAX_BUDGET=4000,PORTFOLIO_SIZE=25,DURATION=18000;
const PRIOR_WIN_RATE=.63,WIN_RATE_MIN=.30,WIN_RATE_MAX=.70;
const ORANGE='#bd784b',GREEN='#2b7457',GOLD='#c8ab63';
const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
const smooth=x=>{x=clamp(x);return x*x*(3-2*x)};
const budgetAt=t=>Math.round(MIN_BUDGET*Math.pow(MAX_BUDGET/MIN_BUDGET,clamp(t)));
const progressAt=k=>Math.log(clamp(k,MIN_BUDGET,MAX_BUDGET)/MIN_BUDGET)/Math.log(MAX_BUDGET/MIN_BUDGET);
const format=k=>k.toLocaleString('en-US');

// One smooth, deterministic synthetic landscape, shared exactly by both panels.
export function qualitySurface(u,v) {
  const g=(x,y,a,b)=>Math.exp(-((u-x)**2/a+(v-y)**2/b));
  return .12+.61*g(.70,.36,.050,.080)+.36*g(.28,.72,.040,.035)+.16*g(.20,.19,.080,.035)-.075*g(.50,.65,.018,.024)+.022*Math.sin(6.283*u)*Math.cos(6.283*v);
}

// A smooth illustrative win-rate trend, scaled near Plot 2's default LightGBM
// comparison (K=64). This is not calculated from the displayed candidates.
export function storyboardQuality(k) {
  const t=progressAt(k);
  return {random:.38+.145*(1-Math.exp(-5.5*t))-.028*smooth((t-.58)/.42),prior:PRIOR_WIN_RATE};
}

function rng(seed) {return()=>{let t=seed+=0x6D2B79F5;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}}
function visible(u,v,z) {
  // Orthographic depth check against the common surface, so points behind a ridge stay hidden.
  const end=Math.min(1-u,1-v),slope=.32/.56;
  for(let d=.015;d<end;d+=.015)if(qualitySurface(u+d,v+d)>z+slope*d+.006)return false;
  return true;
}
function makeCandidates() {
  const random=rng(128409);
  return Array.from({length:MAX_BUDGET},(_,id)=>{
    const u=random(),v=random(),noise=clamp((random()+random()+random()-1.5)*.042,-.055,.055),truth=qualitySurface(u,v),z=truth+noise;
    return {id,u,v,truth,noise,z,visible:visible(u,v,z)};
  });
}
function makePortfolio() {
  // Fixed diagram coordinates. These are not a learned PriorTune artifact or an oracle experiment.
  const coords=[[.60,.32],[.67,.31],[.74,.32],[.80,.38],[.66,.39],[.72,.40],[.76,.46],[.64,.47],[.68,.54],[.79,.53],[.85,.46],[.83,.61],[.60,.58],[.55,.44],[.73,.65],[.24,.68],[.30,.68],[.36,.70],[.23,.76],[.30,.77],[.38,.79],[.36,.87],[.44,.73],[.26,.27],[.35,.29]];
  return coords.map(([u,v],id)=>{const truth=qualitySurface(u,v);return{id,u,v,truth,noise:0,z:truth,visible:visible(u,v,truth)}});
}

function markup() {
  return `<div class="pd-heading"><span class="pd-eyebrow">ILLUSTRATION</span><h2>Random search and a fixed portfolio</h2></div>
  <div class="pd-landscapes">
    <section class="pd-panel pd-random" aria-labelledby="pd-random-title"><div class="pd-panel-title"><h3 id="pd-random-title"><i></i>Random search</h3><span><b id="pd-random-count">4</b> configs</span></div><p class="pd-panel-caption">Keep adding candidates</p><canvas id="pd-random-canvas" role="img" aria-label="A smooth golden hyperparameter surface with a growing set of orange samples; conceptual illustration"></canvas></section>
    <section class="pd-panel pd-prior" aria-labelledby="pd-prior-title"><div class="pd-panel-title"><h3 id="pd-prior-title"><i></i>PriorTune</h3><span><b>25</b> configs</span></div><p class="pd-panel-caption">Reuse a small portfolio</p><canvas id="pd-prior-canvas" role="img" aria-label="The same golden surface with 25 fixed green candidate locations; conceptual illustration"></canvas></section>
  </div>
  <div class="pd-surface-key"><i></i>Same underlying quality surface <span>·</span> Two hyperparameters</div>
  <div class="pd-curve-wrap"><div class="pd-curve-heading"><span>Win rate vs. baseline</span><a class="pd-curve-caveat" href="#random-search">Illustrative trend from Plot 2 ↗</a></div><div id="pd-curve" role="img" aria-label="Illustrative win rate versus a common random-search baseline on the vertical axis, random-search budget on the horizontal axis. Random search rises from about 0.38 to 0.52, then declines slightly to 0.50. The fixed portfolio stays near 0.63. Values approximate the LightGBM trend in Plot 2 and are not measured results."></div></div>
  <div class="pd-controls"><button id="pd-play" class="pd-play" type="button" aria-label="Play illustration" title="Play illustration"><span aria-hidden="true">▶</span></button><div class="pd-slider"><div class="pd-budget-heading"><label for="pd-budget">Random-search budget</label><output id="pd-budget-value" for="pd-budget">4 configurations</output></div><input id="pd-budget" type="range" min="0" max="1000" step="1" value="0" aria-valuetext="4 configurations; logarithmic scale"><div class="pd-budget-ticks"><span>4</span><span>Exponential growth →</span><span>4,000</span></div></div><button id="pd-reset" class="pd-reset" type="button" aria-label="Reset illustration to 4 configurations" title="Reset to 4 configurations"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 5.5A6 6 0 1 1 4 14M4 2v4h4"/></svg></button></div>
  <p id="pd-stage" class="pd-stage" aria-live="polite">A few random candidates cover only a little of the landscape.</p>
  <details class="pd-details"><summary>About this illustration <span>＋</span></summary><ul><li>Both curves show win rate against a common random-search baseline. Their scale follows the LightGBM example in <a href="#random-search">Plot 2</a>: random search levels off near 0.52 and declines slightly, while PriorTune stays near 0.63.</li><li>The surface and win rates here are illustrative. The smooth curve is not calculated from these dots. Plot 2 contains the measured results.</li><li>Both panels use the same underlying quality surface. Orange dots include small noise offsets. Green dots mark the fixed portfolio on the surface, but real validation scores for PriorTune also contain noise.</li></ul></details>`;
}

function createScene(canvas,kind,candidates,portfolio) {
  const ctx=canvas.getContext('2d'),base=document.createElement('canvas');
  let width=0,height=0,dpr=1,project,randomScreen=[],priorScreen=[];
  function rebuild() {
    width=canvas.clientWidth;height=clamp(width*.72+32,155,270);dpr=Math.min(devicePixelRatio||1,2);
    canvas.style.height=`${height}px`;canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
    base.width=canvas.width;base.height=canvas.height;
    const b=base.getContext('2d');b.scale(dpr,dpr);
    const ox=width*.50,oy=height*.66,sx=width*.43,sy=height*.16,sz=height*.56;
    project=(u,v,z)=>({x:ox+(u-v)*sx,y:oy+(u+v-1)*sy-z*sz});
    // A quiet ground plane gives the surface a real three-dimensional frame.
    const corners=[[0,0],[1,0],[1,1],[0,1]].map(([u,v])=>project(u,v,0));
    b.beginPath();corners.forEach((p,i)=>i?b.lineTo(p.x,p.y):b.moveTo(p.x,p.y));b.closePath();b.fillStyle='#f6f4e9';b.fill();
    b.strokeStyle='#e1dfd1';b.lineWidth=.6;
    for(let i=0;i<=4;i++)for(const [a,c] of [[[i/4,0],[i/4,1]],[[0,i/4],[1,i/4]]]){const p=project(...a,0),q=project(...c,0);b.beginPath();b.moveTo(p.x,p.y);b.lineTo(q.x,q.y);b.stroke()}
    const n=34,faces=[];
    for(let i=0;i<n;i++)for(let j=0;j<n;j++){
      const u=(i+.5)/n,v=(j+.5)/n,z=qualitySurface(u,v),dx=(qualitySurface(u+.008,v)-qualitySurface(u-.008,v))/.016,dy=(qualitySurface(u,v+.008)-qualitySurface(u,v-.008))/.016;
      const light=clamp((-.5*dx-.3*dy+.9)/Math.sqrt(dx*dx+dy*dy+1),-.7,1),lum=clamp(73+z*11+light*9,65,92);
      const verts=[[i/n,j/n],[(i+1)/n,j/n],[(i+1)/n,(j+1)/n],[i/n,(j+1)/n]].map(([x,y])=>project(x,y,qualitySurface(x,y)));
      faces.push({depth:i+j,verts,fill:`hsl(43 57% ${lum}%)`});
    }
    faces.sort((a,c)=>a.depth-c.depth).forEach(face=>{b.beginPath();face.verts.forEach((p,i)=>i?b.lineTo(p.x,p.y):b.moveTo(p.x,p.y));b.closePath();b.fillStyle=face.fill;b.fill();b.strokeStyle=face.fill;b.lineWidth=.45;b.stroke()});
    // Sparse curved mesh lines emphasize smoothness without forming a forest of stems.
    b.strokeStyle='#a88c4a60';b.lineWidth=.55;
    for(let i=0;i<=14;i++)for(const flip of [false,true]){
      b.beginPath();let drawing=false;
      for(let j=0;j<=80;j++){
        const u=flip?j/80:i/14,v=flip?i/14:j/80,z=qualitySurface(u,v),p=project(u,v,z);
        if(!visible(u,v,z)){drawing=false;continue}
        drawing?b.lineTo(p.x,p.y):b.moveTo(p.x,p.y);drawing=true;
      }b.stroke();
    }
    const front=project(1,1,0),left=project(0,1,0),right=project(1,0,0);
    b.beginPath();b.moveTo(left.x,left.y);b.lineTo(front.x,front.y);b.lineTo(right.x,right.y);b.strokeStyle='#b8b8a5';b.lineWidth=.8;b.stroke();
    b.fillStyle='#6f7666';b.font=`${width<180?8:9}px system-ui, sans-serif`;b.textBaseline='middle';b.textAlign='center';
    b.fillText('hp₁',(front.x+right.x)*.5+4,(front.y+right.y)*.5+13);
    b.fillText('hp₂',(front.x+left.x)*.5-4,(front.y+left.y)*.5+13);
    randomScreen=candidates.map(p=>({...p,...project(p.u,p.v,p.z)}));priorScreen=portfolio.map(p=>({...p,...project(p.u,p.v,p.z)}));
  }
  function draw(budget) {
    if(!width)return;
    ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(base,0,0);ctx.setTransform(dpr,0,0,dpr,0,0);
    const prior=kind==='prior',points=prior?priorScreen:randomScreen.slice(0,budget),scale=width<190?.76:1;
    const r=prior?3.0*scale:(budget<40?2.45:budget<200?2.0:budget<1000?1.55:1.15)*scale;
    ctx.fillStyle=prior?GREEN:ORANGE;ctx.globalAlpha=prior?1:budget<100?.85:budget<1000?.65:.52;
    if(prior){for(const p of points){if(!p.visible)continue;ctx.beginPath();ctx.arc(p.x,p.y,r+1.1,0,Math.PI*2);ctx.fillStyle='#fffdf1';ctx.fill();ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fillStyle=GREEN;ctx.fill()}}
    else {ctx.beginPath();for(const p of points){if(!p.visible)continue;ctx.moveTo(p.x+r,p.y);ctx.arc(p.x,p.y,r,0,Math.PI*2)}ctx.fill()}
    ctx.globalAlpha=1;
    canvas.dataset.sampleCount=prior?PORTFOLIO_SIZE:budget;canvas.dataset.visibleCount=points.filter(p=>p.visible).length;
  }
  return {rebuild,draw,dimensions:()=>({width,height,dpr})};
}

function makeCurve(el) {
  const w=Math.max(el.clientWidth,230),h=w<400?158:170,m={l:32,r:w<400?62:78,t:13,b:31};
  const x=k=>m.l+progressAt(k)*(w-m.l-m.r),y=q=>h-m.b-(q-WIN_RATE_MIN)/(WIN_RATE_MAX-WIN_RATE_MIN)*(h-m.t-m.b);
  const samples=Array.from({length:161},(_,i)=>{const k=MIN_BUDGET*Math.pow(MAX_BUDGET/MIN_BUDGET,i/160);return {budget:k,...storyboardQuality(k)}});
  const path=arr=>arr.map((p,i)=>`${i?'L':'M'}${x(p.budget)},${y(p.random)}`).join(' ');
  let svg=`<line x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${h-m.b}" class="pd-y-axis"/>`;
  for(const q of [.3,.4,.5,.6,.7])svg+=`<line x1="${m.l}" y1="${y(q)}" x2="${w-m.r}" y2="${y(q)}" class="pd-curve-grid"/><line x1="${m.l-3}" y1="${y(q)}" x2="${m.l}" y2="${y(q)}" class="pd-y-axis"/><text x="${m.l-7}" y="${y(q)+3}" text-anchor="end" class="pd-y-tick">${q.toFixed(1)}</text>`;
  for(const k of [4,25,100,1000,4000])svg+=`<line x1="${x(k)}" y1="${m.t}" x2="${x(k)}" y2="${h-m.b}" class="pd-curve-grid"/><text x="${x(k)}" y="${h-10}" text-anchor="middle">${k>=1000?`${k/1000}k`:k}</text>`;
  svg+=`<path d="${path(samples)}" class="pd-curve-ghost"/><path id="pd-random-line" d="" class="pd-random-line"/><line x1="${x(4)}" y1="${y(PRIOR_WIN_RATE)}" x2="${x(4000)}" y2="${y(PRIOR_WIN_RATE)}" class="pd-prior-line"/><text x="${x(4000)+10}" y="${y(PRIOR_WIN_RATE)+3}" class="pd-curve-prior-label">PriorTune</text><text x="${x(4000)+10}" y="${y(storyboardQuality(4000).random)+3}" class="pd-curve-random-label">Random</text><line id="pd-curve-cursor" x1="${x(4)}" x2="${x(4)}" y1="${m.t}" y2="${h-m.b}" class="pd-curve-cursor"/><circle id="pd-prior-dot" cx="${x(4)}" cy="${y(PRIOR_WIN_RATE)}" r="3.5" class="pd-prior-dot" data-quality="${PRIOR_WIN_RATE}" data-win-rate="${PRIOR_WIN_RATE}"/><circle id="pd-random-dot" cx="${x(4)}" cy="${y(storyboardQuality(4).random)}" r="4" class="pd-random-dot"/>`;
  el.innerHTML=`<svg viewBox="0 0 ${w} ${h}" data-storyboard="true" data-metric="win-rate-vs-baseline" data-x-min="4" data-x-max="4000" data-y-min="${WIN_RATE_MIN}" data-y-max="${WIN_RATE_MAX}" aria-hidden="true">${svg}</svg>`;
  function update(budget) {
    const current=storyboardQuality(budget),xx=x(budget),dot=el.querySelector('#pd-random-dot'),priorDot=el.querySelector('#pd-prior-dot'),cursor=el.querySelector('#pd-curve-cursor');
    dot.setAttribute('cx',xx);dot.setAttribute('cy',y(current.random));dot.dataset.budget=budget;dot.dataset.quality=current.random;dot.dataset.winRate=current.random;
    priorDot.setAttribute('cx',xx);priorDot.dataset.budget=budget;cursor.setAttribute('x1',xx);cursor.setAttribute('x2',xx);
    el.querySelector('#pd-random-line').setAttribute('d',path([...samples.filter(p=>p.budget<budget),{budget,random:current.random}]));
    el.dataset.budget=budget;
  }
  return {update,samples};
}

export function initDemo(root=document.getElementById('prior-demo')) {
  if(!root)return null;
  if(root.dataset.initialized)return window.priortuneDemo;
  root.dataset.initialized='true';root.innerHTML=markup();
  const el=id=>root.querySelector(`#${id}`),candidates=makeCandidates(),portfolio=makePortfolio();
  const randomScene=createScene(el('pd-random-canvas'),'random',candidates,portfolio),priorScene=createScene(el('pd-prior-canvas'),'prior',candidates,portfolio);
  let budget=4,progress=0,playing=false,frame=0,last=0,autoStarted=false,userTouched=false,curve,phase=-1;
  const reducedMotion=window.matchMedia('(prefers-reduced-motion: reduce)');
  const phases=['With few trials, random search misses most of the space.','More trials find better candidates. The portfolio still uses the same 25 configurations.','Trying more configurations now adds little.','At large budgets, a lucky validation score can win over a better configuration.'];
  function render() {
    randomScene.draw(budget);curve?.update(budget);
    el('pd-random-count').textContent=format(budget);el('pd-budget-value').textContent=`${format(budget)} configurations`;
    el('pd-budget').value=Math.round(progress*1000);el('pd-budget').setAttribute('aria-valuetext',`${format(budget)} configurations; logarithmic scale`);
    el('pd-random-canvas').setAttribute('aria-label',`A smooth golden hyperparameter surface with ${format(budget)} orange random samples; conceptual illustration`);
    const next=progress<.18?0:progress<.43?1:progress<.72?2:3;
    if(next!==phase){phase=next;el('pd-stage').textContent=phases[phase];root.dataset.phase=phase;}
    root.dataset.budget=budget;
  }
  function resize() {randomScene.rebuild();priorScene.rebuild();priorScene.draw(PORTFOLIO_SIZE);curve=makeCurve(el('pd-curve'));render()}
  function updatePlay() {el('pd-play').innerHTML=`<span aria-hidden="true">${playing?'Ⅱ':'▶'}</span>`;el('pd-play').setAttribute('aria-label',playing?'Pause illustration':'Play illustration');el('pd-play').title=playing?'Pause illustration':'Play illustration';el('pd-play').setAttribute('aria-pressed',playing);root.dataset.playing=playing;}
  function pause() {playing=false;cancelAnimationFrame(frame);updatePlay()}
  function tick(now) {
    if(!playing)return;
    const delta=Math.min(now-last,100);last=now;progress=clamp(progress+delta/DURATION);budget=budgetAt(progress);render();
    if(progress>=1){pause();return}frame=requestAnimationFrame(tick);
  }
  function play() {if(playing)return;if(progress>=1){progress=0;budget=4;render()}playing=true;last=performance.now();updatePlay();frame=requestAnimationFrame(tick)}
  function setBudget(value) {pause();budget=Math.round(clamp(Number(value)||MIN_BUDGET,MIN_BUDGET,MAX_BUDGET));progress=progressAt(budget);render();return budget}
  function reset() {setBudget(MIN_BUDGET)}
  el('pd-play').addEventListener('click',()=>{userTouched=true;playing?pause():play()});
  el('pd-reset').addEventListener('click',()=>{userTouched=true;reset()});
  el('pd-budget').addEventListener('input',event=>{userTouched=true;pause();progress=clamp(Number(event.target.value)/1000);budget=budgetAt(progress);render()});
  // Keep arrow-key interaction useful even where many log-slider positions round to the same K.
  el('pd-budget').addEventListener('keydown',event=>{
    if(!['ArrowLeft','ArrowDown','ArrowRight','ArrowUp'].includes(event.key))return;
    event.preventDefault();userTouched=true;
    const sign=['ArrowLeft','ArrowDown'].includes(event.key)?-1:1;
    setBudget(sign>0?Math.max(budget+1,Math.round(budget*1.1)):Math.min(budget-1,Math.round(budget/1.1)));
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause()});
  reducedMotion.addEventListener('change',e=>{if(e.matches)pause()});
  let resizeTimer;new ResizeObserver(()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(resize,60)}).observe(root);
  new IntersectionObserver(entries=>{const visible=entries[0].isIntersecting;if(!visible){pause();return}if(!autoStarted&&!userTouched&&!reducedMotion.matches){autoStarted=true;play()}},{threshold:.15}).observe(root);
  resize();updatePlay();
  const api={
    getState:()=>({budget,progress,playing,phase,portfolioSize:PORTFOLIO_SIZE,candidateBankSize:MAX_BUDGET,curve:storyboardQuality(budget),curveMetric:'win rate vs. baseline',curveDomain:[WIN_RATE_MIN,WIN_RATE_MAX],curveKind:'hand-designed schematic; not derived from candidates',curveSamples:curve.samples.map(p=>({...p})),randomSurface:randomScene.dimensions(),priorSurface:priorScene.dimensions(),reducedMotion:reducedMotion.matches}),
    setBudget:value=>{userTouched=true;return setBudget(value)},play:()=>{userTouched=true;play()},pause:()=>{userTouched=true;pause()},reset:()=>{userTouched=true;reset()},surface:qualitySurface,curveAtBudget:storyboardQuality,
    candidates:()=>candidates.map(p=>({...p})),portfolio:()=>portfolio.map(p=>({...p}))
  };
  window.priortuneDemo=api;return api;
}

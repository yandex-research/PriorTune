// The paper's density estimates and exact categorical probability masses.
export function initHyperparameters(data,helpers){
  const {chart,txt,line,path,table,esc,pct,tip,tooltip}=helpers;
  const $=id=>document.getElementById(id),colors={selected:'#2b7457',prior:'#9ba69e'};
  const eligible=data.hyperparameters;
  const chosen=Object.fromEntries(data.models.map(m=>[m.id,m.id==='mlp'?'lr':'learning_rate']));
  const selected=()=>data.models.map(m=>eligible.find(r=>r.model===m.id&&r.key===chosen[m.id]));
  const modelName=r=>data.models.find(m=>m.id===r.model).label;
  const fmt=v=>v===0?'0':Math.abs(v)<.001?Number(v.toPrecision(2)).toExponential().replace('e-0','e-'):Number(v.toPrecision(3)).toLocaleString('en-US',{maximumFractionDigits:6});
  const optionLabel=b=>typeof b.value==='boolean'?(b.value?'On':'Off'):b.label;
  const scope=r=>r.conditional?`${r.nSelected} of ${r.k} selected configurations with TabPrep enabled`:`${r.nSelected} selected configurations`;
  let shuffling=false,activeTip=null;
  function hide(){
    if(activeTip){activeTip.removeAttribute('aria-describedby');activeTip.classList.remove('hp-inspecting');activeTip=null;tooltip.hidden=true}
  }
  function show(node,html,label,x,y){
    if(activeTip&&activeTip!==node){activeTip.removeAttribute('aria-describedby');activeTip.classList.remove('hp-inspecting')}
    activeTip=node;node.classList.add('hp-inspecting');node.setAttribute('aria-describedby',tooltip.id);
    const box=node.getBoundingClientRect();tip(html,x??box.left+box.width/2,y??box.top+box.height/2);
    node.closest('.hp-slot').querySelector('.hp-slot-live').textContent=label;
  }
  document.addEventListener('pointerdown',e=>{if(activeTip&&!e.target.closest('.hp-inspectable'))hide()});
  document.addEventListener('keydown',e=>{if(e.key==='Escape')hide()});
  document.addEventListener('wheel',hide,{passive:true});
  document.addEventListener('touchmove',hide,{passive:true});
  function ticks(lo,hi,log){
    if(!log)return Array.from({length:4},(_,i)=>lo+(hi-lo)*i/3);
    let values=[];
    for(let e=Math.floor(Math.log10(lo));e<=Math.ceil(Math.log10(hi));e++)for(const m of [1,2,5]){const v=m*10**e;if(v>=lo*(1-1e-8)&&v<=hi*(1+1e-8))values.push(v)}
    if(values.length<2)return[lo,hi];
    if(values.length>5)values=values.filter((_,i)=>i%Math.ceil(values.length/4)===0);
    return values;
  }
  function metadata(c,r,distribution){
    Object.assign(c.el.dataset,{key:r.key,kind:r.kind,distribution,nSelected:r.nSelected,conditional:r.conditional,plotTop:c.m.t,plotBottom:c.h-c.m.b,plotLeft:c.m.l,plotRight:c.w-c.m.r});
    c.el.setAttribute('role','group');c.el.setAttribute('aria-label',`${modelName(r)}: ${r.label}. ${scope(r)}.`);
    c.el.querySelector('svg').removeAttribute('aria-hidden');
  }
  function drawDensity(r){
    const mixed=r.kind==='dropout',{points,domain}=r.density,lo=domain[0],hi=domain[1];
    const peak=Math.max(1,...points.map(p=>p.ratio)),step=peak<=2?.5:peak<=5?1:peak<=10?2:5,top=Math.ceil((peak*1.08)/step)*step,yticks=[];
    for(let y=0;y<=top+.001;y+=step)yticks.push(y);
    if(!yticks.includes(1))yticks.push(1);yticks.sort((a,b)=>a-b);
    const c=chart(`hp-${r.model}-chart`,{left:39,right:16,x:domain,y:[0,top],log:r.kind==='log',xticks:ticks(lo,hi,r.kind==='log'),xformat:fmt,yticks,yformat:v=>`${fmt(v)}×`,xlabel:mixed?'Positive dropout rate':r.kind==='log'?'Hyperparameter value · log scale':'Hyperparameter value',ylabel:'Density relative to prior'});
    c.add(`<path class="hp-prior-area" d="M${c.x(lo)},${c.y(0)}L${c.x(lo)},${c.y(1)}L${c.x(hi)},${c.y(1)}L${c.x(hi)},${c.y(0)}Z" fill="${colors.prior}" opacity=".13"/>`);
    c.add(line(c.x(lo),c.y(1),c.x(hi),c.y(1),`class="hp-prior-line" stroke="${colors.prior}" stroke-width="1.5" stroke-dasharray="5 4"`));
    const curve=path(points,c.x,c.y,'x','ratio');
    c.add(`<path d="${curve} L${c.x(hi)},${c.y(0)} L${c.x(lo)},${c.y(0)} Z" fill="${colors.selected}" opacity=".045"/>`);
    c.add(`<path class="data-line hp-density-line" d="${curve}" stroke="${colors.selected}"/>`);
    r.selected.filter(v=>!mixed||v>0).forEach(v=>c.add(line(c.x(v),c.y(0)+4,c.x(v),c.y(0)+9,'class="hp-rug" stroke="#2b7457" stroke-width=".7" opacity=".32"')));
    c.add(`<circle class="hp-inspection-dot" cx="${c.x(points[150].x)}" cy="${c.y(points[150].ratio)}" r="3.5"/>`);
    c.finish();metadata(c,r,mixed?'mixed':'density');
    Object.assign(c.el.dataset,{xMin:lo,xMax:hi,yMax:top,xScale:r.density.xScale});
    c.el.setAttribute('aria-label',`${modelName(r)}: ${r.label}. ${mixed?'Positive-rate ':''}density relative to the sampling prior. ${scope(r)}.${mixed?' The probability of exactly zero appears separately below.':''}`);
    const node=c.el.querySelector('svg');let index=150,pinned=false;
    node.classList.add('hp-inspectable','hp-density-control');node.setAttribute('tabindex','0');node.setAttribute('role','slider');
    node.setAttribute('aria-label',`${modelName(r)}: inspect ${r.label.toLowerCase()} density. Use left and right arrow keys.`);
    node.setAttribute('aria-valuemin',lo);node.setAttribute('aria-valuemax',hi);
    function value(){const p=points[index];return{p,label:`${r.label}: ${fmt(p.x)}. Selected density ${p.ratio.toFixed(2)} times the sampling prior.`}}
    function inspect(x,y){
      const {p,label}=value();node.setAttribute('aria-valuenow',p.x);node.setAttribute('aria-valuetext',label);
      const dot=node.querySelector('.hp-inspection-dot');dot.setAttribute('cx',c.x(p.x));dot.setAttribute('cy',c.y(p.ratio));
      show(node,`<strong>${esc(r.label)} · ${fmt(p.x)}</strong><br>Selected density: ${p.ratio.toFixed(2)}× prior<br><small>${esc(modelName(r))} · ${esc(scope(r))}${mixed?'<br>Positive rates only; zero mass is separate.':''}</small>`,label,x,y);
    }
    node.setAttribute('aria-valuenow',points[index].x);node.setAttribute('aria-valuetext',value().label);
    function pointer(e){const b=node.getBoundingClientRect(),u=Math.max(0,Math.min(1,((e.clientX-b.left)*c.w/b.width-c.m.l)/(c.w-c.m.l-c.m.r)));index=Math.round(u*(points.length-1));inspect(e.clientX,e.clientY)}
    node.addEventListener('pointermove',e=>{if(e.pointerType==='mouse'){pinned=false;pointer(e)}});
    node.addEventListener('click',e=>{pinned=true;pointer(e)});
    node.addEventListener('pointerleave',()=>{if(!pinned&&document.activeElement!==node)hide()});
    node.addEventListener('focus',()=>inspect());node.addEventListener('blur',()=>{pinned=false;hide()});
    node.addEventListener('keydown',e=>{
      const step=e.shiftKey?10:1;
      if(['ArrowLeft','ArrowDown','ArrowRight','ArrowUp','Home','End'].includes(e.key)){
        e.preventDefault();index=e.key==='Home'?0:e.key==='End'?points.length-1:Math.max(0,Math.min(points.length-1,index+(['ArrowLeft','ArrowDown'].includes(e.key)?-step:step)));pinned=true;inspect();
      }else if(e.key==='Escape'){pinned=false;hide()}
    });
  }
  function bindMass(node,r,b){
    let pinned=false;
    const label=`${r.label}, ${optionLabel(b)}: sampling prior ${pct(b.prior,2)}; selected ${pct(b.selected,2)}, ${b.count} of ${r.nSelected} configurations.${r.conditional?' Conditional on TabPrep being enabled.':''}`;
    const html=`<strong>${esc(r.label)} · ${esc(optionLabel(b))}</strong><br>Prior: ${pct(b.prior,2)}<br>Selected: ${pct(b.selected,2)} · ${b.count}/${r.nSelected}<br><small>${esc(modelName(r))}${r.conditional?' · TabPrep enabled':''}</small>`;
    node.classList.add('hp-inspectable');node.setAttribute('aria-label',label);
    node.addEventListener('pointermove',e=>{if(e.pointerType==='mouse'){pinned=false;show(node,html,label,e.clientX,e.clientY)}});
    node.addEventListener('pointerleave',()=>{if(!pinned&&document.activeElement!==node)hide()});
    node.addEventListener('click',e=>{pinned=true;show(node,html,label,e.clientX||undefined,e.clientY||undefined)});
    node.addEventListener('focus',()=>show(node,html,label));node.addEventListener('blur',()=>{pinned=false;hide()});
    node.addEventListener('keydown',e=>{
      if(e.key==='Enter'||e.key===' '){e.preventDefault();pinned=true;show(node,html,label)}
      else if(e.key==='Escape'){pinned=false;hide()}
      else if(e.key==='ArrowRight'||e.key==='ArrowLeft'){
        const peers=[...node.parentElement.querySelectorAll('.hp-category')],i=peers.indexOf(node);if(i>=0){e.preventDefault();peers[(i+(e.key==='ArrowRight'?1:-1)+peers.length)%peers.length].focus()}
      }
    });
  }
  function drawCategorical(r){
    const bins=r.bins,peak=Math.max(...bins.flatMap(b=>[b.prior,b.selected])),top=Math.min(1,Math.ceil((peak+.025)*10)/10),step=top<=.4?.1:.2,yticks=[];
    for(let y=0;y<top-1e-8;y+=step)yticks.push(y);yticks.push(top);
    const c=chart(`hp-${r.model}-chart`,{left:39,right:16,x:[-.5,bins.length-.5],y:[0,top],xticks:bins.map((_,i)=>i),xformat:i=>optionLabel(bins[i]),yticks,yformat:v=>pct(v,0),xlabel:'Hyperparameter choice',ylabel:'Share of configurations'});
    const groupWidth=(c.w-c.m.l-c.m.r)/bins.length,barWidth=Math.min(25,groupWidth*.30),gap=Math.min(4,groupWidth*.05);
    bins.forEach((b,i)=>{
      c.add(`<g class="hp-category" tabindex="0" role="button" data-bin-index="${i}" data-prior="${b.prior}" data-selected="${b.selected}" data-count="${b.count}" data-n-selected="${r.nSelected}">`);
      c.add(`<rect class="hp-category-target" x="${c.x(i)-groupWidth*.46}" y="${c.m.t}" width="${groupWidth*.92}" height="${c.h-c.m.t-c.m.b+27}" rx="3" fill="transparent"/>`);
      [['prior',b.prior,-barWidth-gap/2],['selected',b.selected,gap/2]].forEach(([series,share,offset])=>c.add(`<rect class="hp-bar hp-${series}-bar" data-series="${series}" data-share="${share}" ${series==='selected'?`data-count="${b.count}"`:''} x="${c.x(i)+offset}" y="${c.y(share)}" width="${barWidth}" height="${c.y(0)-c.y(share)}" rx="1.5" fill="${colors[series]}"/>`));
      c.add('</g>');
    });c.finish();metadata(c,r,'categorical');
    Object.assign(c.el.dataset,{xMin:-.5,xMax:bins.length-.5,yMax:top,xScale:'categorical'});
    const node=c.el.querySelector('svg');node.setAttribute('role','group');node.setAttribute('aria-label',`${r.label}: prior and selected shares`);
    node.querySelectorAll('.hp-category').forEach((n,i)=>bindMass(n,r,bins[i]));
  }
  function drawAtom(r,panel){
    const atom=panel.querySelector('.hp-atom');atom.innerHTML='';atom.hidden=r.kind!=='dropout';if(atom.hidden)return;
    const b=r.bins.find(b=>b.value===0);
    atom.innerHTML=`<button type="button" class="hp-zero-mass" data-prior="${b.prior}" data-selected="${b.selected}" data-count="${b.count}" data-n-selected="${r.nSelected}"><span class="hp-atom-heading">Exactly zero <span>${b.count}/${r.nSelected} selected</span></span>${[['prior','Prior',b.prior],['selected','Selected',b.selected]].map(([series,label,share])=>`<span class="hp-atom-row"><span>${label}</span><span class="hp-atom-track"><i class="hp-atom-bar hp-${series}-bar" data-series="${series}" data-share="${share}" style="width:${share*100}%"></i></span><b>${pct(share,0)}</b></span>`).join('')}</button>`;
    bindMass(atom.querySelector('button'),r,b);
  }
  function draw(r){
    hide();const panel=$(`hp-slot-${r.model}`);panel.querySelector('select').value=r.key;
    Object.assign(panel.dataset,{key:r.key,kind:r.kind,nSelected:r.nSelected,conditional:r.conditional});
    if(r.kind==='cat'||r.kind==='prep')drawCategorical(r);else drawDensity(r);
    drawAtom(r,panel);
    const kind=r.kind==='log'?'Log-uniform prior · 1×':r.kind==='lin'?'Uniform prior · 1×':r.kind==='dropout'?'Positive rates + zero mass':'Prior and selected shares';
    panel.querySelector('.hp-slot-note').innerHTML=`<span>${r.conditional?`${r.nSelected}/${r.k} selected · TabPrep enabled`:r.kind==='dropout'?`${r.density.nKernelSamples} positive / ${r.nSelected} selected`:`${r.nSelected} selected configurations`}</span><span>${kind}</span>`;
  }
  function details(){
    $('hp-note').textContent='Lines show density relative to the prior. Bars show shares. Marks below a curve are selected configurations.';
    $('hp-protocol').innerHTML='<ul><li>Continuous curves use the paper’s reflected density estimate. A value of 1× matches the original sampling prior.</li><li>Categorical bars show shares and counts. TabPrep-only choices use the 37 selected configurations with TabPrep enabled.</li><li>The chance of exactly zero dropout is shown separately from the density of positive rates.</li><li>These distributions describe the selected portfolio. They are not the probability that a value is optimal.</li></ul>';
    const rows=selected().flatMap(r=>{
      const prefix=[modelName(r),r.label];
      if(r.kind==='cat'||r.kind==='prep')return r.bins.map(b=>[...prefix,optionLabel(b),'Share',pct(b.prior,2),pct(b.selected,2),`${b.count}/${r.nSelected}${r.conditional?' · TabPrep enabled':''}`]);
      const density=r.density.points.map(p=>[...prefix,fmt(p.x),'Density / prior','1×',`${p.ratio.toFixed(4)}×`,'—']);
      if(r.kind==='dropout'){const b=r.bins.find(b=>b.value===0);density.unshift([...prefix,'Exactly zero','Share',pct(b.prior,2),pct(b.selected,2),`${b.count}/${r.nSelected}`])}
      return density;
    });
    table('hp-table',['Model','Hyperparameter','Value','Measure','Prior','Selected','Count'],rows,'Current slots: exact category shares and counts, or 300 paper-derived density samples. The download retains selected values, original histogram bins and estimator metadata.');
  }
  function render(){
    if(!$('hp-chart').children.length){
      $('hp-chart').innerHTML=data.models.map(m=>`<section id="hp-slot-${m.id}" class="small-multiple hp-slot" data-model="${m.id}"><div class="panel-title"><h4>${esc(m.label)}</h4></div><div class="hp-reel"><label class="hp-parameter"><span class="sr-only">${esc(m.label)} hyperparameter</span><select id="hp-key-${m.id}" aria-label="${esc(m.label)} hyperparameter">${eligible.filter(r=>r.model===m.id).map(r=>`<option value="${esc(r.key)}">${esc(r.label)}</option>`).join('')}</select></label><div id="hp-${m.id}-chart" class="chart panel-chart"></div><div class="hp-atom" hidden></div><div class="hp-slot-note"></div><span class="hp-slot-live sr-only" role="status"></span></div></section>`).join('');
      data.models.forEach(m=>$(`hp-key-${m.id}`).addEventListener('change',e=>{chosen[m.id]=e.target.value;draw(selected().find(r=>r.model===m.id));details()}));
    }
    selected().forEach(draw);details();
  }
  $('hp-shuffle').addEventListener('click',async()=>{
    if(shuffling)return;shuffling=true;hide();
    const button=$('hp-shuffle'),reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
    button.disabled=true;button.classList.add('pulling');
    try{
      await Promise.all(data.models.map(async(m,i)=>{
        const options=eligible.filter(r=>r.model===m.id&&r.key!==chosen[m.id]);
        const r=options[Math.floor(Math.random()*options.length)],reel=$(`hp-slot-${m.id}`).querySelector('.hp-reel');
        if(!reduced)await reel.animate([{opacity:1,transform:'translateY(0)'},{opacity:0,transform:'translateY(-16px)'}],{duration:150,delay:i*65,easing:'ease-in',fill:'forwards'}).finished.then(()=>reel.getAnimations().forEach(a=>a.cancel()));
        chosen[m.id]=r.key;draw(r);
        if(!reduced)await reel.animate([{opacity:0,transform:'translateY(18px)'},{opacity:1,transform:'translateY(0)'}],{duration:260,easing:'cubic-bezier(.2,.7,.25,1)'}).finished;
      }));
      details();$('hp-announcement').textContent=selected().map(r=>`${modelName(r)}: ${r.label}`).join('. ');
    }finally{button.disabled=false;button.classList.remove('pulling');shuffling=false}
  });
  render();return{render,selected};
}

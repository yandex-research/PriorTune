/* BeyondArena figures. Display filters never refit the published 58-method Elo field. */
const COLORS = {'lgbm':'#5887aa','lgbm-prep':'#2b7457','mlp':'#8070af','tabpack':'#be794d'};
const TEXT_COLORS = {'lgbm':'#47728f','lgbm-prep':'#2b7457','mlp':'#706094','tabpack':'#9e603b'};
const FAMILY_NAMES = {'lgbm':'LightGBM','lgbm-prep':'LightGBM + TabPrep','mlp':'MLP','tabpack':'TabPack'};
const PFNS = ['kumo-tabular','kumo-tabular-medium','kumo-tabular-small','tabpfn-3','tabiclv2'];
const CLASSICS = ['realmlp','tabm','catboost'];
const MODE_NAMES = {recompute:'Recompute',cache_gpu:'KV cache · GPU',cache_cpu:'KV cache · CPU'};
const MODE_COLORS = {recompute:'#e5e8e6',cache_gpu:'#515b55',cache_cpu:'#a5ada7'};
const $ = id => document.getElementById(id);
const esc = v => String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v,d=1) => Number(v).toLocaleString('en-US',{maximumFractionDigits:d});
const text = (x,y,label,attrs='') => `<text x="${x}" y="${y}" ${attrs.replace(/fill="([^"]+)"/g,'style="fill:$1"')}>${esc(label)}</text>`;
const line = (x1,y1,x2,y2,attrs='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}/>`;
const color = r => COLORS[r.family] || '#858e88';
const textColor = r => TEXT_COLORS[r.family] || '#5a645d';
const regimeLabel = r => r === 'TE' ? 'T+E' : r;
const isOwn = r => Object.hasOwn(COLORS,r.family);

function include(r,regime) {
  if(regime==='all') return true;
  // The compact comparison follows Figure 3's FT/T and FTE/T+E pairings.
  // TabPack is always the published 64-member default pack, never relabeled T+E.
  if(r.family==='tabpack') return r.regime==='D' && ['R','FTE'].includes(r.arm);
  if(isOwn(r)) return r.regime===regime && ['R',regime==='T'?'FT':'FTE'].includes(r.arm);
  if(PFNS.includes(r.family)) return r.regime==='D';
  return CLASSICS.includes(r.family) && r.regime===regime;
}

function name(r,full=false) {
  let s=FAMILY_NAMES[r.family] || r.name;
  s=s.replace('Kumo-Tabular-Medium','Kumo · medium').replace('Kumo-Tabular-Small','Kumo · small').replace('Kumo-Tabular','Kumo-Tabular').replace('TabICLv2','TabICL v2');
  if(isOwn(r)) s+=` ${r.arm}`;
  if(full) s+=` · ${regimeLabel(r.regime)}`;
  else if(r.family==='tabpack') s+=' · 64';
  return s;
}

function tooltip(r,latency=false) {
  const selection=r.arm==='R'?'Random portfolio':r.arm==='FT'?'Portfolio selected for tuning':r.arm==='FTE'?'Portfolio selected for ensembling':'Published reference';
  return `<strong>${esc(r.label)}</strong><br>${fmt(r.elo)} Elo${latency?` · ${fmt(r.latencyMs,3)} ms`:''}<br><small>${selection}${isOwn(r)?` · K = ${r.family==='tabpack'?64:25}`:''}<br>95% Elo interval ${fmt(r.ciLow)}–${fmt(r.ciHigh)}${latency?`<br>${esc(MODE_NAMES[r.mode]||'Native serving')}<br>${esc(r.hardware)}<br>Latency: 10 datasets · Elo: 142 datasets`:''}</small>`;
}

function marker(r,x,y,id,latency=false,size=5.4) {
  const c=color(r), cache=Boolean(r.mode), s=size;
  let d;
  if(cache) d=`M${x} ${y-s-1} L${x+s+1} ${y} L${x} ${y+s+1} L${x-s-1} ${y} Z`;
  else if(r.regime==='T') d=`M${x-s} ${y-s} h${2*s} v${2*s} h${-2*s} Z`;
  else if(r.regime==='TE') {
    const a=s*.46;
    d=`M${x-s} ${y-s+a} L${x-s+a} ${y-s} L${x} ${y-a} L${x+s-a} ${y-s} L${x+s} ${y-s+a} L${x+a} ${y} L${x+s} ${y+s-a} L${x+s-a} ${y+s} L${x} ${y+a} L${x-s+a} ${y+s} L${x-s} ${y+s-a} L${x-a} ${y} Z`;
  } else d=`M${x-s} ${y} a${s} ${s} 0 1 0 ${2*s} 0 a${s} ${s} 0 1 0 ${-2*s} 0`;
  const fill=cache?MODE_COLORS[r.mode]:r.arm==='R'||r.arm==='FT'?'#fff':c;
  const info=tooltip(r,latency);
  const attrs=`data-method="${esc(r.method)}" data-family="${esc(r.family)}" data-arm="${esc(r.arm)}" data-regime="${r.regime}" data-elo="${r.elo}" data-x="${x}" data-y="${y}"${latency?` data-latency-ms="${r.latencyMs}" data-mode="${r.mode}"`:''}`;
  const half=r.arm==='FT'?`<defs><clipPath id="${id}"><rect x="${x-s-2}" y="${y-s-2}" width="${s+2}" height="${2*s+4}"/></clipPath></defs><path d="${d}" fill="${c}" clip-path="url(#${id})"/>`:'';
  return `<g class="ba-mark" ${attrs} tabindex="0" role="graphics-symbol" aria-label="${esc(`${r.label}, ${fmt(r.elo)} Elo${latency?`, ${fmt(r.latencyMs,3)} milliseconds, ${MODE_NAMES[r.mode]||'native serving'}`:''}`)}" data-tip="${esc(info)}"><title>${esc(`${r.label}: ${fmt(r.elo)} Elo${latency?`, ${fmt(r.latencyMs,3)} ms; ${MODE_NAMES[r.mode]||'native serving'}`:''}`)}</title><circle cx="${x}" cy="${y}" r="11" fill="transparent"/><path d="${d}" fill="${fill}" stroke="${cache?'#6d766f':c}" stroke-width="1.7"/>${half}${half?`<path d="${d}" fill="none" stroke="${c}" stroke-width="1.7"/>`:''}</g>`;
}

function legend(latency,regime) {
  const arms=regime==='all'?'<span><i class="ba-symbol ba-half"></i>FT · tune-selected</span><span><i class="ba-symbol ba-full"></i>FTE · ensemble-selected</span>':`<span><i class="ba-symbol ${regime==='T'?'ba-half':'ba-full'}"></i>PriorTune · ${regime==='T'?'FT':'FTE'}</span>`;
  return `<div class="ba-key"><div class="ba-key-main"><span><i class="ba-symbol ba-empty"></i>Random portfolio</span>${arms}${regime==='all'?'<span class="ba-regime-key"><b>○</b> D <b>□</b> T <b>×</b> T+E</span>':''}${latency?'<span class="ba-frontier-key"><i></i>Displayed frontier</span>':'<span class="ba-ci-key"><i></i>95% interval</span>'}</div>${latency?'<div class="ba-cache-key"><span>In-context serving:</span><span><i class="ba-diamond ba-cache-gpu"></i>GPU cache</span><span><i class="ba-diamond ba-cache-cpu"></i>CPU cache</span><span><i class="ba-diamond ba-recompute"></i>Recompute</span></div>':''}</div>`;
}

function rankPlot(rows,regime,width) {
  const all=regime==='all';
  let groups;
  if(all) groups=rows.map(r=>({name:name(r,true),rows:[r],pair:false}));
  else {
    const seen=new Set(); groups=[];
    for(const r of rows) {
      if(isOwn(r)) {
        if(seen.has(r.family)) continue;
        seen.add(r.family);
        groups.push({name:FAMILY_NAMES[r.family],rows:rows.filter(v=>v.family===r.family),pair:true});
      } else groups.push({name:name(r),rows:[r],pair:false});
    }
    groups.sort((a,b)=>Math.max(...b.rows.map(r=>r.elo))-Math.max(...a.rows.map(r=>r.elo)));
  }
  const narrow=width<600&&!all,w=all?Math.max(width,650):Math.max(width,250);
  const m={l:all?235:narrow?(w<300?110:w<400?120:160):190,r:all?68:narrow?(w<300?55:63):135,t:34,b:47}, step=all?29:narrow?43:39,h=m.t+groups.length*step+m.b;
  const lo=Math.floor((Math.min(...rows.map(r=>r.ciLow))-10)/50)*50, hi=Math.ceil((Math.max(...rows.map(r=>r.ciHigh))+10)/50)*50;
  const x=v=>m.l+(v-lo)/(hi-lo)*(w-m.l-m.r);
  let svg='';
  const tickStep=w-m.l-m.r<160?200:100;
  for(let v=Math.ceil(lo/tickStep)*tickStep;v<=hi;v+=tickStep) svg+=line(x(v),m.t-14,x(v),h-m.b,'class="ba-grid"')+text(x(v),h-m.b+23,fmt(v,0),'text-anchor="middle" class="ba-tick"');
  svg+=text(m.l,14,'BEYONDARENA ELO ↑','class="ba-axis-heading"')+text(w-5,14,all?'ELO':'ELO / GAIN','class="ba-axis-heading" text-anchor="end"');
  groups.forEach((g,i)=>{
    const y=m.t+i*step+step*.5;
    const selected=g.rows.find(r=>r.isSelected), base=g.rows.find(r=>r.arm==='R');
    if(g.pair) svg+=`<rect x="2" y="${y-step/2+3}" width="${w-4}" height="${step-6}" rx="3" fill="${color(g.rows[0])}" opacity=".045"/>`;
    const labelColor=g.pair?textColor(g.rows[0]):all&&isOwn(g.rows[0])?textColor(g.rows[0]):'#445148';
    const label=narrow?g.name.replace('LightGBM + TabPrep','LGBM + TabPrep'):g.name;
    svg+=text(m.l-(narrow?8:14),y+(g.pair?0:4),label,`text-anchor="end" class="ba-row-name${g.pair?' ba-strong':''}" fill="${labelColor}"`);
    if(g.pair) svg+=text(m.l-(narrow?8:14),y+14,g.rows[0].family==='tabpack'?'64-member pack · D':`${regimeLabel(regime)} · 25 configurations`,`text-anchor="end" class="ba-row-sub"`);
    if(selected&&base) svg+=line(x(base.elo),y+4,x(selected.elo),y-4,`stroke="${color(selected)}" stroke-width="3" opacity=".5"`);
    // Intervals are vertically offset inside a paired row so both remain inspectable.
    g.rows.slice().sort((a,b)=>Number(a.isSelected)-Number(b.isSelected)).forEach((r,j)=>{
      const yy=y+(g.pair?(r.isSelected?-4:4):0), c=color(r);
      svg+=line(x(r.ciLow),yy,x(r.ciHigh),yy,`stroke="${c}" stroke-width="1.3" opacity="${r.isSelected?.55:.32}"`);
      svg+=line(x(r.ciLow),yy-2,x(r.ciLow),yy+2,`stroke="${c}" opacity=".5"`)+line(x(r.ciHigh),yy-2,x(r.ciHigh),yy+2,`stroke="${c}" opacity=".5"`);
      svg+=marker(r,x(r.elo),yy,`ba-rank-${i}-${j}`,false,r.isSelected?5.4:4.5);
    });
    if(selected&&base) {
      svg+=text(w-(narrow?4:60),y+(narrow?-2:4),fmt(selected.elo,0),'text-anchor="end" class="ba-value ba-strong"');
      const delta=selected.elo-base.elo;
      svg+=text(w-5,y+(narrow?13:4),`${delta>=0?'+':''}${fmt(delta)}`,`text-anchor="end" class="ba-gain" fill="${textColor(selected)}"`);
    } else svg+=text(w-5,y+4,fmt(g.rows[0].elo,0),'text-anchor="end" class="ba-value"');
  });
  svg+=text((m.l+w-m.r)/2,h-3,narrow?'Higher is better':'Higher is better · published Elo stays fixed when filtering','text-anchor="middle" class="ba-axis-label"');
  return `<div class="ba-scroll ba-rank-scroll${narrow?' ba-rank-narrow':''}" tabindex="0" aria-label="Elo chart${all?'; scroll horizontally on small screens':''}"><svg class="ba-rank-svg" viewBox="0 0 ${w} ${h}" style="min-width:${w}px" role="img" aria-label="BeyondArena Elo, ${rows.length} methods" data-x-min="${lo}" data-x-max="${hi}" data-plot-left="${m.l}" data-plot-right="${w-m.r}" data-method-count="${rows.length}">${svg}</svg></div>`;
}

// Text labels are placed outside markers; measured values are never visually jittered.
function placeLabels(items,points,bounds,occupied=[]) {
  const boxes=occupied.map(b=>({...b})),out=[];
  const overlap=(a,b)=>Math.max(0,Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x))*Math.max(0,Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y));
  for(const item of items) {
    const ww=item.label.length*6.05+9,hh=17, candidates=[];
    for(const dy of [-15,15,-32,32,-49,49,-66,66,-83,83,0]) {
      for(const dx of [12,-ww-12,28,-ww-28,56,-ww-56,90,-ww-90]) {
        const box={x:Math.max(bounds.l,Math.min(bounds.r-ww,item.x+dx)),y:Math.max(bounds.t,Math.min(bounds.b-hh,item.y+dy-hh/2)),w:ww,h:hh};
        let cost=Math.abs(box.x+ww*.5-item.x)*.055+Math.abs(box.y+hh*.5-item.y)*.8;
        cost+=boxes.reduce((s,b)=>s+overlap(box,b)*15,0);
        cost+=points.reduce((s,p)=>s+overlap(box,{x:p.x-8,y:p.y-8,w:16,h:16})*6,0);
        candidates.push({box,cost});
      }
    }
    candidates.sort((a,b)=>a.cost-b.cost);
    const box=candidates[0].box; boxes.push({...box,x:box.x-3,w:box.w+6,y:box.y-1,h:box.h+2});out.push({...item,box});
  }
  return out;
}

// A shared label column gives nearly coincident points room to be read. The
// ordering follows Elo, so leaders do not reorder methods or move data markers.
function columnLabels(items,left,top,bottom) {
  const sorted=[...items].sort((a,b)=>a.y-b.y),gap=23;
  const centers=sorted.map((p,i)=>Math.max(p.y,i?0:top));
  for(let i=1;i<centers.length;i++)centers[i]=Math.max(centers[i],centers[i-1]+gap);
  if(centers.at(-1)>bottom){centers[centers.length-1]=bottom;for(let i=centers.length-2;i>=0;i--)centers[i]=Math.min(centers[i],centers[i+1]-gap)}
  return sorted.map((p,i)=>({...p,column:true,box:{x:left,y:centers[i]-8.5,w:p.label.length*6.05+9,h:17}}));
}

function latencyPlot(rows,regime,width,overview=false,detail=false) {
  const all=regime==='all',w=overview?Math.max(width,250):Math.max(width,all?1080:900),h=overview?380:all?760:555;
  const m=overview?{l:37,r:16,t:30,b:49}:{l:58,r:29,t:37,b:56};
  const min=Math.min(...rows.map(r=>r.latencyMs)),max=Math.max(...rows.map(r=>r.latencyMs));
  const lo=Math.pow(10,Math.floor(Math.log10(min))),hi=Math.pow(10,Math.ceil(Math.log10(max)));
  const ymin=Math.floor((Math.min(...rows.map(r=>r.elo))-28)/50)*50,ymax=Math.ceil((Math.max(...rows.map(r=>r.elo))+22)/50)*50;
  const x=v=>m.l+(Math.log10(v)-Math.log10(lo))/(Math.log10(hi)-Math.log10(lo))*(w-m.l-m.r),y=v=>h-m.b-(v-ymin)/(ymax-ymin)*(h-m.t-m.b);
  let svg='';
  const yStep=overview?100:50;
  for(let v=Math.ceil(ymin/yStep)*yStep;v<=ymax;v+=yStep) svg+=line(m.l,y(v),w-m.r,y(v),'class="ba-grid"')+text(m.l-(overview?7:11),y(v)+4,fmt(v,0),'text-anchor="end" class="ba-tick"');
  for(let power=Math.floor(Math.log10(lo));power<=Math.ceil(Math.log10(hi));power++) {
    for(const multiple of [1,2,5]) {
      const v=multiple*10**power;if(v<lo||v>hi)continue;
      if(!overview||multiple===1) svg+=line(x(v),m.t,x(v),h-m.b,`class="ba-grid${multiple===1?'':' ba-minor-grid'}"`);
      if(multiple===1||(!all&&multiple===5&&w>1000)) svg+=text(x(v),h-m.b+23,v>=1000?`${v/1000}k`:fmt(v,v<1?3:0),'text-anchor="middle" class="ba-tick"');
    }
  }
  svg+=text(m.l,15,overview?'ELO ↑':'BEYONDARENA ELO ↑','class="ba-axis-heading"')+text(w-m.r,15,overview?'← FASTER':'← FASTER INFERENCE','class="ba-axis-heading" text-anchor="end"');
  const points=rows.map(r=>({r,x:x(r.latencyMs),y:y(r.elo)}));
  const best=[];let highest=-Infinity;
  for(const p of [...points].sort((a,b)=>a.r.latencyMs-b.r.latencyMs||b.r.elo-a.r.elo)) if(p.r.elo>highest){best.push(p);highest=p.r.elo;}
  if(best.length) {
    let d=`M${best[0].x} ${best[0].y}`;
    best.slice(1).forEach(p=>{d+=` H${p.x} V${p.y}`});d+=` H${w-m.r}`;
    svg+=`<path d="${d}" class="ba-frontier"/>`;
  }
  // A horizontal link groups one PFN's cache modes; its published Elo is held fixed.
  const cacheGroups=new Map();
  for(const p of points.filter(p=>p.r.mode)) {if(!cacheGroups.has(p.r.method))cacheGroups.set(p.r.method,[]);cacheGroups.get(p.r.method).push(p);}
  for(const ps of cacheGroups.values()) svg+=line(Math.min(...ps.map(p=>p.x)),ps[0].y,Math.max(...ps.map(p=>p.x)),ps[0].y,'class="ba-cache-link"');
  if(!all)for(const family of Object.keys(COLORS)) {
    const pair=points.filter(p=>p.r.family===family),a=pair.find(p=>p.r.arm==='R'),b=pair.find(p=>p.r.isSelected);
    if(a&&b)svg+=line(a.x,a.y,b.x,b.y,`stroke="${color(a.r)}" class="ba-pair-link"`);
  }
  const labelItems=[];
  for(const ps of cacheGroups.values()) {
    const p=ps.reduce((a,b)=>a.x<b.x?a:b),right=Math.max(...ps.map(q=>q.x));
    labelItems.push({r:p.r,x:overview?p.x:(p.x+right)/2,y:p.y,label:name(p.r,all),color:'#5a645d',own:false,method:p.r.method,members:[p.r.method]});
  }
  for(const family of Object.keys(COLORS)) {
    const group=points.filter(p=>p.r.family===family);if(!group.length)continue;
    const p=group.reduce((a,b)=>a.r.elo>b.r.elo?a:b);
    labelItems.push({...p,label:FAMILY_NAMES[family]+(family==='tabpack'?' · 64':''),color:textColor(p.r),own:true,method:p.r.method,members:group.map(v=>v.r.method)});
  }
  for(const p of points.filter(p=>!p.r.mode&&!isOwn(p.r)))labelItems.push({...p,label:name(p.r,all),color:textColor(p.r),own:false,method:p.r.method,members:[p.r.method]});
  const items=overview?labelItems.filter(p=>['kumo-tabular','tabpfn-3'].includes(p.r.family)||(p.r.family==='lgbm-prep'&&p.r.arm===(regime==='T'?'FT':'FTE')&&p.r.regime===(all?'TE':regime))||(p.r.family==='tabpack'&&p.r.arm==='FTE')):labelItems;
  if(overview)for(const p of items)p.label=p.label.replace('LightGBM + TabPrep','TabPrep').replace(' · 64','').replace(' · T+E','').replace(' · D','');
  let labels;
  if(overview)labels=placeLabels(items,points,{l:m.l+3,r:w-m.r-3,t:m.t+3,b:h-m.b-4});
  else {
    const conventional=items.filter(p=>!p.r.mode),cache=items.filter(p=>p.r.mode);
    const left=Math.max(...points.filter(p=>!p.r.mode).map(p=>p.x))+29;
    const column=columnLabels(conventional,left,m.t+15,h-m.b-13);
    const cacheLabels=placeLabels(cache,points,{l:m.l+3,r:w-m.r-3,t:m.t+3,b:h-m.b-4},column.map(p=>({...p.box,x:p.box.x-5,w:p.box.w+10,y:p.box.y-3,h:p.box.h+6})));
    labels=[...column,...cacheLabels];
  }
  labels.forEach(p=>{
    const bx=Math.max(p.box.x,Math.min(p.box.x+p.box.w,p.x)),by=Math.max(p.box.y,Math.min(p.box.y+p.box.h,p.y));
    if(p.column){const end=p.box.x-5,cy=p.box.y+p.box.h/2;svg+=`<path d="M${p.x+7} ${p.y} C${p.x+28} ${p.y} ${end-25} ${cy} ${end} ${cy}" stroke="${p.color}" class="ba-column-leader" data-leader-method="${esc(p.method)}"/>`}
    else if(Math.hypot(bx-p.x,by-p.y)>7)svg+=line(p.x,p.y,bx,by,`stroke="${p.color}" class="ba-label-leader"`);
  });
  points.forEach((p,i)=>svg+=marker(p.r,p.x,p.y,`ba-lat-${overview?'mini':'full'}-${i}`,true,overview?(p.r.isSelected?3.8:3.2):p.r.isSelected?6:5.3).replace('class="ba-mark"',`class="${detail?'ba-detail-mark':'ba-mark'}"`));
  labels.forEach(p=>svg+=`<g class="ba-direct-label${p.column?' ba-column-label':''}" data-label-method="${esc(p.method)}" data-label-family="${esc(p.r.family)}" data-label-members="${esc(JSON.stringify(p.members))}"><rect x="${p.box.x-1}" y="${p.box.y}" width="${p.box.w+2}" height="${p.box.h}" rx="2" fill="#fff" opacity=".94"/>${text(p.box.x+4,p.box.y+12.5,p.label,`fill="${p.color}" class="${p.own?'ba-strong':''}"`)}</g>`);
  svg+=text((m.l+w-m.r)/2,h-5,overview?'Inference · ms, log scale':'Single-row inference · milliseconds (log scale)','text-anchor="middle" class="ba-axis-label"');
  return `<div class="ba-scroll ba-latency-scroll${overview?' ba-overview':''}" tabindex="0" aria-label="${overview?'Latency overview; all measurements shown':'Labeled latency chart; scroll horizontally on small screens'}"><svg class="ba-latency-svg" viewBox="0 0 ${w} ${h}" style="min-width:${w}px" role="img" aria-label="Elo versus single-row inference time; ${rows.length} measurements" data-secondary="${detail}" data-x-min="${lo}" data-x-max="${hi}" data-y-min="${ymin}" data-y-max="${ymax}" data-plot-left="${m.l}" data-plot-right="${w-m.r}" data-plot-top="${m.t}" data-plot-bottom="${h-m.b}" data-method-count="${rows.length}">${svg}</svg></div>`;
}

function renderTable(rows,latency) {
  const headers=latency?['Method','Elo','Inference (ms)','Serving mode','Hardware']:['Method','Elo','95% interval'];
  $('elo-table').innerHTML=`<table><caption>Every method shown above. Values are published results; display filters do not refit Elo.</caption><thead><tr>${headers.map(s=>`<th scope="col">${s}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${(latency?[r.label,fmt(r.elo),fmt(r.latencyMs,3),MODE_NAMES[r.mode]||'Native serving',r.hardware]:[r.label,fmt(r.elo),`${fmt(r.ciLow)}–${fmt(r.ciHigh)}`]).map(s=>`<td>${esc(s)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

export function renderBeyondArena(data,view='rank') {
  const regime=$('elo-regime').value, latency=view==='latency', all=regime==='all';
  const rows=(latency?data.latency:data.elo).filter(r=>include(r,regime));
  const el=$('elo-chart');el.classList.add('ba-figure');el.style.height='auto';el.setAttribute('role','group');
  el.setAttribute('aria-label',latency?'BeyondArena accuracy and inference time':'BeyondArena accuracy comparison');
  const width=el.clientWidth;
  const mobileLatency=latency&&width<600;
  const plot=mobileLatency?latencyPlot(rows,regime,width,true)+`<details class="ba-labeled-detail"><summary>Explore the fully labeled figure <span>＋</span></summary><p class="ba-scroll-hint">Scroll horizontally to compare all methods →</p>${latencyPlot(rows,regime,width,false,true)}</details>`:latency?latencyPlot(rows,regime,width):rankPlot(rows,regime,width);
  el.innerHTML=legend(latency,regime)+(!mobileLatency&&((latency&&width<900)||(all&&width<650))?'<p class="ba-scroll-hint">Scroll horizontally to compare all methods →</p>':'')+plot;
  const scope=$('elo-scope');if(scope)scope.textContent=latency?'Elo · 142 datasets / timing · 10 datasets':`142 datasets · ${rows.length} of 58 methods`;
  $('elo-note').textContent=latency?'Single-row inference. Colored labels identify model families. Hover for the portfolio and deployment setting. Linked diamonds show cache modes of the same model.':all?'All 58 methods, with published 95% intervals. FT/FTE denote portfolio selection. D/T/T+E denote deployment.':'Each colored pair compares PriorTune with a random portfolio of the same model. The right column shows Elo and the gain. TabPack contains 64 configurations.';
  const protocol=[
    'Elo uses 142 BeyondArena datasets, 497 tasks and eight inner folds. Scores are fitted on the full 58-method field and stay fixed when the display is filtered.',
    'Inference timing uses 10 datasets: 50 single-row calls after 5 warm-ups, taking the median within each dataset and then across datasets. Trees use 16 CPU threads. Neural and in-context models use an H100.',
    'FT selects a portfolio for tuning. FTE selects one for ensembling. T deploys one validation-selected configuration. T+E ensembles configurations. Portfolios contain 25 configurations, except the 64-member TabPack at its published default setting.',
    all?'All methods includes 58 Elo results and 47 available timing measurements.':'The selected comparison shows FT with T, FTE with T+E, and TabPack FTE/R at D. References are Kumo (three sizes), TabPFN-3, TabICL v2, RealMLP, TabM and CatBoost. All methods includes the remaining references.',
    'Cache timings share the same published model Elo. The dashed line shows the accuracy–latency frontier among displayed measurements. Every marker is plotted at its exact measured time.'
  ];
  $('elo-protocol').textContent=protocol.join('\n\n');$('elo-protocol').classList.add('ba-protocol');
  renderTable(rows,latency);
}

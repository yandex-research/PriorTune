#!/usr/bin/env python3
"""Check the conceptual demo and touch interactions without running experiments."""
import argparse
import json
import math
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:4175/')
    parser.add_argument('--browser')
    parser.add_argument('--output',type=Path,default=ROOT/'.qa/experience')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    checks=[];errors=[]
    def check(value,name,evidence=None):checks.append({'pass':bool(value),'check':name,'evidence':evidence})
    with sync_playwright() as p:
        options={'headless':True}
        chrome=Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
        if args.browser or chrome.is_file():options['executable_path']=args.browser or str(chrome)
        browser=p.chromium.launch(**options)
        context=browser.new_context(viewport={'width':1440,'height':1000})
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(args.url,wait_until='networkidle')
        page.wait_for_function('document.documentElement.dataset.ready === "true"')
        page.evaluate('window.priortuneDemo.pause()')
        initial=page.evaluate('({state:window.priortuneDemo.getState(),candidates:window.priortuneDemo.candidates(),portfolio:window.priortuneDemo.portfolio()})')
        check(len(initial['candidates'])==4000 and len(initial['portfolio'])==25,'Illustration has a fixed 4,000-candidate bank and 25 diagram coordinates')
        check(page.evaluate('window.priortuneDemo.candidates().concat(window.priortuneDemo.portfolio()).every(p=>Math.abs(p.truth-window.priortuneDemo.surface(p.u,p.v))<1e-12)'),
              'Both panels use the same underlying quality surface')
        visible=page.locator('#prior-demo').inner_text().lower()
        check('illustration' in visible and 'win rate vs. baseline' in visible and 'illustrative trend' in visible,
              'Illustrative status and the curve units are visible without opening details')
        disclosure=page.locator('.pd-details').text_content().lower()
        check('not calculated from' in disclosure and 'validation' in disclosure and 'noise' in disclosure,
              'Demo disclosure avoids experimental and noiseless-validation claims')
        ticks=page.locator('#pd-curve .pd-y-tick').all_text_contents()
        check(ticks==['0.3','0.4','0.5','0.6','0.7'] and page.locator('#pd-curve .pd-y-axis').count()>=1,
              'Demo curve has a numeric win-rate axis covering 0.3 to 0.7',ticks)
        link=page.locator('.pd-curve-heading a[href="#random-search"]')
        check(link.count()==1 and 'Plot 2' in link.inner_text()
              and page.locator('#random-search').count()==1,
              'Curve links directly to the measured random-search figure')
        samples=initial['state']['curveSamples'];random=[r['random'] for r in samples]
        check(all(r['prior']>r['random'] for r in samples) and random[0]<max(random) and 0<max(random)-random[-1]<.1,
              'Designed trajectory rises, gently declines, and stays below its fixed reference')
        check(initial['state']['curveMetric']=='win rate vs. baseline'
              and initial['state']['curveDomain']==[.3,.7]
              and math.isclose(random[0],.38,abs_tol=1e-12)
              and .519<max(random)<.521 and .496<random[-1]<.497
              and all(math.isclose(r['prior'],.63,abs_tol=1e-12) for r in samples),
              'Illustrative win rates use the intended common baseline and scale',
              {'start':random[0],'peak':max(random),'final':random[-1],'prior':samples[0]['prior']})
        prior_pixels=page.locator('#pd-prior-canvas').evaluate('c=>c.toDataURL()')
        for budget in (4,25,100,1000,4000):
            page.evaluate('(k)=>window.priortuneDemo.setBudget(k)',budget)
            state=page.evaluate('window.priortuneDemo.getState()')
            dot=page.locator('#pd-random-dot').evaluate('n=>({...n.dataset})')
            check(state['budget']==budget and int(page.locator('#pd-random-canvas').get_attribute('data-sample-count'))==budget
                  and int(page.locator('#pd-prior-canvas').get_attribute('data-sample-count'))==25,
                  f'Budget changes random sample count while the portfolio stays fixed: {budget}')
            check(int(dot['budget'])==budget and math.isclose(float(dot['quality']),state['curve']['random'],abs_tol=1e-12)
                  and math.isclose(float(dot['winRate']),state['curve']['random'],abs_tol=1e-12)
                  and int(page.locator('#pd-curve').get_attribute('data-budget'))==budget,
                  f'Current curve marker agrees with the displayed schematic budget: {budget}')
            check(page.locator('#pd-prior-canvas').evaluate('c=>c.toDataURL()')==prior_pixels,
                  f'Prior surface and fixed points remain visually unchanged: {budget}')
            if budget in (4,100,4000):
                page.locator('#prior-demo').screenshot(path=str(args.output/f'1440-demo-{budget}.png'),style='header,.skip-link{visibility:hidden!important}')
        check(page.evaluate('window.priortuneDemo.candidates()')==initial['candidates'] and page.evaluate('window.priortuneDemo.portfolio()')==initial['portfolio'],
              'Changing budgets preserves both fixed coordinate sets')
        for identifier in ('pd-random-canvas','pd-prior-canvas'):
            pixels=page.locator('#'+identifier).evaluate('c=>{const d=c.getContext("2d").getImageData(0,0,c.width,c.height).data;let n=0;for(let i=3;i<d.length;i+=4)if(d[i])n++;return n}')
            check(pixels>1000,'Smooth surface is rendered: '+identifier,pixels)
        page.locator('#pd-budget').fill('0')
        check(page.evaluate('window.priortuneDemo.getState().budget')==4,'Native budget slider reaches four candidates')
        page.locator('#pd-budget').fill('1000')
        check(page.evaluate('window.priortuneDemo.getState().budget')==4000,'Native budget slider reaches 4,000 candidates')
        page.locator('#pd-reset').click()
        page.locator('#pd-play').click();page.wait_for_timeout(500)
        check(page.evaluate('window.priortuneDemo.getState().progress')>0,'Play advances the schematic story')
        page.locator('#pd-play').click();paused=page.evaluate('window.priortuneDemo.getState().progress');page.wait_for_timeout(200)
        check(page.evaluate('window.priortuneDemo.getState().progress')==paused,'Pause freezes playback')
        page.locator('#pd-reset').click()
        check(page.evaluate('window.priortuneDemo.getState().budget')==4,'Reset returns the linked surfaces and curve to the beginning')
        context.close()
        for width in (390,320):
            context=browser.new_context(viewport={'width':width,'height':844},has_touch=True,is_mobile=True,reduced_motion='reduce',device_scale_factor=1)
            page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(args.url,wait_until='networkidle');page.wait_for_function('document.documentElement.dataset.ready === "true"')
            page.wait_for_timeout(200)
            check(not page.evaluate('window.priortuneDemo.getState().playing'),'Reduced motion suppresses automatic demo playback at '+str(width)+'px')
            page.evaluate('window.priortuneDemo.setBudget(4000)')
            page.locator('#prior-demo').screenshot(path=str(args.output/f'{width}-demo.png'),style='header,.skip-link{visibility:hidden!important}')
            sizes=page.locator('#prior-demo canvas').evaluate_all('nodes=>nodes.map(n=>({width:n.clientWidth,height:n.clientHeight,samples:+n.dataset.sampleCount}))')
            check(len(sizes)==2 and all(r['width']>=100 and r['height']>=150 for r in sizes),'Both surfaces remain usable on mobile: '+str(width),sizes)
            page.locator('#heatmap-k [data-k="25"]').tap()
            for model,n,c in [('lgbm',25,25),('mlp',4000,4000)]:
                cell=page.locator(f'.heatmap-cell[data-model="{model}"][data-n="{n}"][data-c="{c}"]')
                cell.tap();tip=page.locator('#chart-tooltip')
                box=tip.bounding_box()
                check(tip.is_visible() and cell.get_attribute('aria-describedby')=='chart-tooltip',f'Tap immediately exposes cell details: {width}/{model}/{n}/{c}')
                check(box['x']>=0 and box['y']>=0 and box['x']+box['width']<=width+1 and box['y']+box['height']<=845,
                      f'Touch tooltip stays within the viewport: {width}/{model}',box)
                page.screenshot(path=str(args.output/f'{width}-touch-{model}.png'),style='header,.skip-link{visibility:hidden!important}')
                page.touchscreen.tap(3,400)
                check(tip.is_hidden(),'Tapping outside dismisses tile details: '+str(width)+'/'+model)
            page.locator('#heatmap-k [data-k="64"]').tap()
            check(page.locator('#heatmap-chart .heatmap-cell').count()==675,'Touch K control updates all measured grids: '+str(width))
            before=page.locator('.hp-slot').evaluate_all('nodes=>nodes.map(n=>n.dataset.key)')
            page.locator('#hp-shuffle').tap();page.wait_for_function('!document.getElementById("hp-shuffle").disabled')
            after=page.locator('.hp-slot').evaluate_all('nodes=>nodes.map(n=>n.dataset.key)')
            check(all(a!=b for a,b in zip(before,after)) and len(after)==3,'Touch lever changes all slots with reduced motion: '+str(width))
            for model,key in [('lgbm','extra_trees'),('lgbm-prep','oafe_max_feats'),('mlp','dropout')]:
                page.locator('#hp-key-'+model).select_option(key)
            for model in ('lgbm','lgbm-prep'):
                bar=page.locator('#hp-'+model+'-chart .hp-category').first
                bar.tap();tip=page.locator('#chart-tooltip');box=tip.bounding_box()
                check(tip.is_visible() and ('109/200' if model=='lgbm' else '19/37') in tip.text_content(),
                      'Touch bar details preserve exact conditional counts: '+str(width)+'/'+model,tip.text_content())
                check(box['x']>=0 and box['x']+box['width']<=width+1 and box['y']>=0 and box['y']+box['height']<=845,
                      'Categorical touch details stay on screen: '+str(width)+'/'+model,box)
                page.touchscreen.tap(3,400)
                check(tip.is_hidden(),'Outside tap dismisses categorical details: '+str(width)+'/'+model)
            atom=page.locator('#hp-slot-mlp .hp-zero-mass');atom.tap()
            check(page.locator('#chart-tooltip').is_visible() and '116/200' in page.locator('#chart-tooltip').text_content(),
                  'Touch dropout zero mass shows its own count: '+str(width))
            page.touchscreen.tap(3,400)
            page.locator('#inside').screenshot(path=str(args.output/f'{width}-hp-mixed.png'),style='header,.skip-link,#chart-tooltip{visibility:hidden!important}')
            ticks=page.locator('#pd-curve .pd-y-tick').evaluate_all('nodes=>nodes.map(n=>({text:n.textContent,width:n.getBoundingClientRect().width}))')
            check(len(ticks)==5 and all(t['width']>0 for t in ticks),'Numeric demo y-axis remains visible on mobile: '+str(width),ticks)
            check(page.evaluate('document.documentElement.scrollWidth<=document.documentElement.clientWidth'),'No horizontal page overflow during mobile interactions: '+str(width))
            context.close()
        check(not errors,'No JavaScript errors throughout demo and touch interactions',errors)
        browser.close()
    report={'url':args.url,'passed':sum(c['pass'] for c in checks),'total':len(checks),'checks':checks}
    (args.output/'experience-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f"{report['passed']}/{report['total']} experience checks passed; report: {args.output}")
    for row in checks:
        if not row['pass']:print('FAIL:',row['check'],row['evidence'])
    raise SystemExit(0 if all(c['pass'] for c in checks) else 1)

if __name__=='__main__':main()

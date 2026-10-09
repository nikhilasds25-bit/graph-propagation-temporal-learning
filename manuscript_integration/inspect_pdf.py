"""Render and inspect the compiled manuscript; read-only PDF QA."""
from pathlib import Path
import json,re,subprocess,sys
from PIL import Image,ImageDraw
import pdfplumber
BASE=Path(__file__).resolve().parent
PDF=BASE/'final_manuscript/main.pdf'
suffix='_final' if '--final' in sys.argv else ''
OUT=BASE/('qa/pdf_pages'+suffix);OUT.mkdir(exist_ok=True)
POPPLER=Path(r'C:\Users\Nikhil\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe')
subprocess.run([str(POPPLER),'-r','110','-png',str(PDF),str(OUT/'page')],check=True,capture_output=True)
pages=sorted(OUT.glob('page-*.png'))
alltext=[];report=[]
with pdfplumber.open(PDF) as doc:
    for n,p in enumerate(doc.pages,1):
        t=p.extract_text() or '';alltext.append(t)
        bounds=[c for c in p.chars if c.get('text','').strip() and (c['x0']<-1 or c['x1']>p.width+1 or c['top']<-1 or c['bottom']>p.height+1)]
        report.append({'page':n,'width':p.width,'height':p.height,'rotation':p.rotation,'characters_outside_page':len(bounds),
                       'text_preview':t[:180],'contains_forecast_table':bool('Brier score' in t)})
for start in range(0,len(pages),6):
    sheet=Image.new('RGB',(1530,1490),'#d0d0d0');draw=ImageDraw.Draw(sheet)
    for j,p in enumerate(pages[start:start+6]):
        im=Image.open(p).convert('RGB');im.thumbnail((485,700))
        x=(j%3)*510;y=(j//3)*745
        sheet.paste(im,(x+(510-im.width)//2,y+30))
        draw.text((x+10,y+6),f'Page {start+j+1}',fill='black')
    sheet.save(BASE/'qa'/f'pdf_montage{suffix}_{start//6+1}.png')
text='\n'.join(alltext)
checks={'page_count':len(report),'references_heading_count':sum(len(re.findall(r'^References$',t,re.M)) for t in alltext),
        'outside_page_characters':sum(p['characters_outside_page'] for p in report),
        'unresolved_reference_markers':'??' in text,'pages':report}
(BASE/'qa/pdf_text.txt').write_text(text,encoding='utf-8')
(BASE/'qa/pdf_layout_audit.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in checks.items() if k!='pages'},indent=2))
print('Forecast table pages:',[p['page'] for p in report if p['contains_forecast_table']])

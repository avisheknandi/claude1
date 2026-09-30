"""Fetch primary bibliographic metadata; retain provenance and reject weak title matches."""
import difflib, html, json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests

HERE=Path(__file__).resolve().parent
JOURNAL=[
('halabi','The RSNA Pediatric Bone Age Machine Learning Challenge','10.1148/radiol.2018180736'),
('larson','Performance of a Deep-Learning Neural Network Model in Assessing Skeletal Maturity on Pediatric Hand Radiographs','10.1148/radiol.2017170236'),
('spampinato','Deep learning for automated skeletal bone age assessment in X-ray images','10.1016/j.media.2016.10.010'),
('lee','Fully Automated Deep Learning System for Bone Age Assessment',None),
('chen','Attention-Guided Discriminative Region Localization and Label Distribution Learning for Bone Age Assessment','10.1109/JBHI.2021.3095128'),
('bonetplus','A two-stream network with global-local feature fusion for bone age assessment','10.1038/s41598-026-54051-9'),
('rfga','Region fine-grained attention network for accurate bone age assessment','10.3934/mbe.2024081'),
('stage','A developmental stage-aware graph transformer framework for automated bone-age assessment',None),
('cfjl','CFJLNet: Coarse and Fine Feature Joint Learning Network for Bone Age Assessment','10.1109/TIM.2022.3193711'),
('selfsupervised','Self-Supervised Attention Mechanism for Pediatric Bone Age Assessment With Efficient Weak Annotation','10.1109/TMI.2020.3046672'),
('bogff','Coarse-to-Fine bone age regression by using multi-scale self-attention mechanism','10.1016/j.bspc.2024.107029'),
('lightweight','Hand bone age estimation using divide and conquer strategy and lightweight convolutional neural networks','10.1016/j.engappai.2023.105935'),
('ensemble','Improving Automated Pediatric Bone Age Estimation Using Ensembles of Models from the 2017 RSNA Machine Learning Challenge','10.1148/ryai.2019190053'),
('stress','Evaluating the Robustness of a Deep Learning Bone Age Algorithm to Clinical Image Variation Using Computational Stress Testing','10.1148/ryai.230240'),
('bias','Generalizability and Bias in a Deep Learning Pediatric Bone Age Prediction Model Using Hand Radiographs','10.1148/radiol.220505'),
('integral','Integral Bone Age Regression','10.1016/j.procs.2024.09.373'),
('bonet','Hand pose estimation for pediatric bone age assessment',None),
('bonexpert','The BoneXpert method for automated determination of skeletal maturity',None),
]
ARXIV=[
('iglovikov','1712.05053'),('mobilenetv3','1905.02244'),('mobilenetv2','1801.04381'),
('resnet','1512.03385'),('efficientnet','1905.11946'),('transformer','1706.03762'),
('vit','2010.11929'),('swin','2103.14030'),('gradcache','2101.06983'),
('adamw','1711.05101'),('gradcam','1610.02391'),('mixed','1710.03740'),
('checkpoint','1604.06174'),('calibration','1706.04599'),('uncertainty','1703.04977'),
('coral','1901.07884'),('unet','1505.04597'),('pytorch','1912.01703'),
('squeeze','1709.01507'),('batchnorm','1502.03167'),('layernorm','1607.06450'),
('cbam','1807.06521')]

def clean(s): return re.sub(r'[^a-z0-9]','',html.unescape(s).lower())

def journal_record(row):
    key,title,doi=row
    if doi:
        url='https://api.crossref.org/works/'+doi
        r=requests.get(url,timeout=45);r.raise_for_status();item=r.json()['message']
    else:
        url='https://api.crossref.org/works'
        r=requests.get(url,params={'query.title':title,'rows':5},timeout=45);r.raise_for_status()
        items=r.json()['message']['items']
        item=max(items,key=lambda a:difflib.SequenceMatcher(None,clean(title),clean(a.get('title',[''])[0])).ratio())
    score=difflib.SequenceMatcher(None,clean(title),clean(item.get('title',[''])[0])).ratio()
    if score<.8: raise ValueError(f'Weak metadata match {key}: {score}: {item.get("title")}')
    date=item.get('published-print',item.get('published',item.get('issued')))['date-parts'][0]
    return dict(key=key,title=html.unescape(item['title'][0]),authors=[(a.get('given','')+' '+a.get('family','')).strip() for a in item.get('author',[])],
                year=date[0],venue=html.unescape(item.get('container-title',[''])[0]),doi=item['DOI'],url='https://doi.org/'+item['DOI'],
                volume=item.get('volume',''),pages=item.get('page',item.get('article-number','')),verified_title_similarity=score,
                provenance=url,accessed='2026-09-30')

def arxiv_record(row):
    key,identifier=row;url='https://arxiv.org/abs/'+identifier
    r=requests.get(url,timeout=45);r.raise_for_status();body=r.text
    def metas(name):
        return [html.unescape(v) for v in re.findall(r'<meta\s+name="'+name+r'"\s+content="([^"]+)"',body)]
    title=metas('citation_title');authors=metas('citation_author');date=metas('citation_date')
    if not title or not authors or not date: raise ValueError(f'Metadata missing {identifier}')
    return dict(key=key,title=title[0],authors=authors,year=int(date[0][:4]),venue='arXiv preprint arXiv:'+identifier,
                doi='10.48550/arXiv.'+identifier,url=url,provenance=url,accessed='2026-09-30')

def tex_escape(s):
    replacements={'&':r'\&','%':r'\%','_':r'\_','#':r'\#'}
    return ''.join(replacements.get(c,c) for c in s)

def main():
    records=[];errors=[]
    # Limited concurrency for public metadata services.
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks=[('journal',x) for x in JOURNAL]+[('arxiv',x) for x in ARXIV]
        def work(task):
            kind,row=task
            for attempt in range(3):
                try: return (journal_record(row) if kind=='journal' else arxiv_record(row)),None
                except Exception as ex:
                    if attempt==2: return None,dict(key=row[0],error=str(ex))
                    time.sleep(2+attempt*3)
        for record,error in pool.map(work,tasks):
            if error: errors.append(error);print(error,flush=True)
            else: records.append(record);print(record['key'],record['title'],flush=True)
    (HERE/'references_verified.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')
    (HERE/'references_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
    if len(records)<36: raise RuntimeError(f'Only {len(records)} verified references; resolve metadata failures')
    bib=[r'\begin{thebibliography}{99}']
    for r in records:
        authors=', '.join(r['authors'][:3])+(' et al.' if len(r['authors'])>3 else '')
        detail=r.get('volume','')
        if r.get('pages'): detail+=', pp. '+str(r['pages'])
        bib.append(r'\bibitem{'+r['key']+'} '+tex_escape(authors)+', ``'+tex_escape(r['title'])+'", '+r'\emph{'+tex_escape(r['venue'])+'}, '+tex_escape(detail)+', '+str(r['year'])+'. '+r'\url{'+r['url']+'}')
    bib.append(r'\end{thebibliography}')
    (HERE/'paper'/'references.tex').write_text('\n'.join(bib),encoding='utf-8')
    print('VERIFIED',len(records),flush=True)

if __name__=='__main__': main()

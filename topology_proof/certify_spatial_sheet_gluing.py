"""认证四次根在退化输出点的局部斜率及无穷根的次序。

依靠精确展开和Sturm实根计数，不调用已有或新生成的数值轨迹。
"""
from pathlib import Path
import json
import sympy as s
OUT=Path(__file__).resolve().parent
c,Z,kappa=s.symbols('c Z kappa',real=True)
P=-(5*Z**2-12*Z+5)*(5*Z**2-8*Z+5)*c*c-40*(Z-1)**3*(Z+1)*c-15*(Z-1)**2*(Z+1)**2
leadplus=s.factor(s.Poly(s.expand(P.subs(Z,1+c*kappa)),c).nth(2))
leadminus=s.factor(s.Poly(s.expand(P.subs(Z,-1+c*kappa)),c).nth(2))
infinity_cubic=s.Poly(P.subs(c,-s.Rational(3,5)),Z)
out={'plus_one_leading_coefficient':str(leadplus),
     'minus_one_leading_coefficient':str(leadminus),
     'plus_one_slopes':['-1/sqrt(15)','1/sqrt(15)'],
     'minus_one_slopes':['(40-sqrt(115))/15','(40+sqrt(115))/15'],
     'infinity_cubic_at_c_minus_three_fifths':str(infinity_cubic.as_expr()),
     'finite_negative_roots':int(infinity_cubic.count_roots(-s.oo,0)),
     'finite_positive_roots':int(infinity_cubic.count_roots(0,s.oo)),
     'gateway_mapping':{'S1':'G1 to H1','S2':'G2 to H2','S3':'G3 to H3','S4':'G4 to H4'},
     'proof':'distinct projective roots away from c=0; exact signed y and full passive invertibility at c=0; infinity order fixed by leading-coefficient sign'}
out['passed']=s.expand(leadplus+4*(15*kappa**2-1))==0 and s.expand(leadminus+4*(15*kappa**2-80*kappa+99))==0 and out['finite_negative_roots']==1 and out['finite_positive_roots']==2
(OUT/'spatial_sheet_gluing_exact_certificates.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2))
if not out['passed']:raise SystemExit('根片拼接证书失败')

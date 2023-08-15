# function script for the redcap train
import os.path
import requests
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from redcap import Project
from fpdf import FPDF
from datetime import datetime

transformation_dict = {'HTRA1-related autosomal dominant cerebral small vessel disease': 'HTRA1',
                       'X-linked Adrenoleukodystrophy; ORPHA:139399': 'X-ALD',
                       'Peroxisome biogenesis disorder ; ORPHA:79189': 'Peroxisome biogenesis disorder',
                       'Zellweger syndrome	; ORPHA:912': 'Zellweger syndrome',
                       'Morbus Refsum	; ORPHA:772': 'Morbus Refsum',
                       'α-Methylacyl-CoA racemase deficiency	; ORPHA:79095': 'α-Methylacyl-CoA',
                       'Metachromatic leukodystrophy	; ORPHA:512': 'Meachromatic LD',
                       'Krabbe disease	; ORPHA:487': 'Krabbe disease',
                       'Multiple Sulfatase Deficiency	; ORPHA:585': 'Multiple Sulfatase',
                       'Salla disease	; ORPHA:309334': 'Salla disease',
                       'Fucosidosis	; ORPHA:349': 'Fucosidosis',
                       'VPS11-related hypomyelination	; ORPHA:466934': 'VPS11',
                       'Kearns Sayre syndrome (KSS)	; ORPHA:480': 'KSS',
                       'Complex I defects	; ORPHA: 2609 u.a.': 'Complex I',
                       'Complex II defects	; ORPHA: 3208': 'Complex II',
                       'Complex III defects	; ORPHA: 1460': 'Complex III',
                       'Complex IV defects	; ORPHA: 254905': 'Complex III',
                       'Iron sulfur cluster defects	; ORPHA:457406': 'Fe sulfur cluster',
                       'Pyruvate carboxylase deficiency	; ORPHA:3008': 'Pyruvate carboxylase',
                       'Mitochondrial neuro-gastrointestinal encephalopathy (MNGIE)	; ORPHA:298': 'MNGIE',
                       'Canavan disease	; ORPHA:141': 'Canavan disease',
                       'L-2-hydroxyglutaric aciduria	; ORPHA:79314': 'L-2-hydroxyglutaric aciduria',
                       'Maple syrup urine disease (MSUD)	; ORPHA:268145': 'MSUD',
                       'Phenylketonuria (PKU)	; ORPHA:716': 'PKU',
                       '3-Hydroxy-3-methylglutaryl-CoA (HMGCoA) lyase deficiency	; ORPHA:35701': 'HMGCoA',
                       'Defects in cobalamin, homocysteine and folate metabolism	; ORPHA:79282': 'cobalamin,..., meta',
                       'Cerebral folate transport deficiency	; ORPHA:217382': 'Cerebral folate',
                       'GABA transaminase deficiency	; ORPHA:2066': 'GABA',
                       'Vanishing white matter (VWM)	; ORPHA:135': 'VWM',
                       '4H syndrome	; ORPHA:289494': '4H syndrome',
                       'Leukodystrophy with brainstem and spinal cord involvement and lactate elevation (LBSL)	; '
                       'ORPHA:137898': 'LBSL',
                       'Leukodystrophy with thalamus and brainstem cord involvement and lactatem elevation (LTBL)	; '
                       'ORPHA:314051': 'LTBL',
                       'AARS2-related leukodystrophy	; ORPHA:313808': 'AARS2',
                       'Hypomyelination with brainstem and spinal cord involvement and leg spasticity (HBSL)	; '
                       'ORPHA:363412': 'HBSL',
                       'RARS-related hypomyelination	; ORPHA:438114': 'RARS',
                       'EPRS-related hypomyelination': 'EPRS',
                       'AIMP1-related hypomyelination	; ORPHA:280293': 'AIMP1',
                       'Megalencephalic leukodystrophy with subcortical cysts (MLC)	; ORPHA:2478': 'MLC',
                       'CLCN2-related leukodystrophy	; ORPHA:363540': 'CLCN2',
                       'X-linked Charcot-Marie-Tooth disease (CMTX)	; ORPHA:64747': 'CMTX',
                       'Congenital muscular dystrophies': 'Congenital muscular dystrophies',
                       'GPR56-related leukodystrophy': 'GPR56',
                       'MYRF-related leukodystrophy': 'MYRF',
                       'Pelizaeus-Merzbacher disease (PMD)	; ORPHA:702': 'PMD',
                       'Pelizaeus-Merzbacher-like disease	; ORPHA:280270': 'PMD like',
                       'Oligodentodigital dysplasia (ODDD)	; ORPHA:2710': 'ODDD',
                       'Oculodentodigital dysplasia (ODDD)\t; ORPHA:2710': 'ODDD',
                       'Cerebral autosomal dominant arteriopathy met subcortical infarcts and leukodystrophy ('
                       'CADASIL)	; ORPHA:136': 'CADASIL',
                       'Cerebral autosomal recessive arteriopathy met subcortical infarcts and leukodystrophy ('
                       'CARASIL)	; ORPHA:199354': 'CARASIL',
                       'Cathepsin A-related arteriopathy with strokes and leukodystrophy (CARASAL)	; ORPHA:575553':
                           'CARASAL',
                       'Hereditary cerebral amyloid angiopathy	; ORPHA:439254': 'cerebral amyloid angiopathy',
                       'Fabry disease	; ORPHA:324': 'Fabry disease',
                       'COL4A1-/COL4A2-related disease	; ORPHA:477759': 'COL4A1/2',
                       'Retinal vasculopathy and cerebral leukodystrophy (RVCL)	; ORPHA:247691': 'RVCL',
                       'Aicardi-Goutières syndrome	; ORPHA:51': 'Aicardi-Goutières',
                       'RNASET2-related leukodystrophy	; ORPHA:85136': 'RNASET2',
                       'Leukodystrophy with calcifications and cysts (LCC)	; ORPHA:542310': 'LCC',
                       'Coates Plus	; ORPHA:313838': 'Coates Plus',
                       'Cockayne syndrome	; ORPHA:191': 'Cockayne syndrome',
                       'Trichothiodystrophy with hypomyelination	; ORPHA:33364': 'Trichothiodystrophy',
                       'Alexander disease	; ORPHA:58': 'Alexander disease',
                       'adult polyglucosan body disease	; ORPHA:206583': 'adult polyglucosan body',
                       'Giant axonal neuropathy	; ORPHA:643': 'Giant axonal neuropathy',
                       'Porphyria-related leukodystrophy	; ORPHA: 79276': 'Porphyria',
                       'Cerebrotendinous xanthomatosis	; ORPHA:909': 'Cerebrotendinous xanthomatosis',
                       'Incontinentia pigmenti (IP)	; ORPHA:464': 'IP',
                       'Sjögren-Larssen syndrome	; ORPHA:816': 'Sjögren-Larssen syndrome',
                       'Lowe oculocerebrorenal syndrome	; ORPHA:534': 'Lowe',
                       'Fragile X tremor/ataxia syndrome	; ORPHA:93256': 'X tremor/ataxia',
                       'Adult onset dominant leukodystrophy (ADLD)	; ORPHA:99027': 'ADLD',
                       'Adult onset leukodystrophy with axonal spheroids and pigmented glia (ALSP) due to CSF1 R '
                       'mutation; ORPHA:313808': 'ALSP-CSF1',
                       'Nasu Hakola disease; ORPHA:2770': 'Nasu Hakola',
                       'Dentatorubropallidoluysian atrophy (DRPLA)	; ORPHA:101': 'DRPLA',
                       'Hypomyelination with atrophy of the basal ganglia and cerebellum (HABC) / TUBB4A-related '
                       'hypomyelination	; ORPHA:139441': 'HABC',
                       'TMEM106B-related hypomyelination': 'THRM106B',
                       'X-linked hypomyelination with spondylometaphyseal dysplasia	; ORPHA:83629':
                           'spondylometaphyseal dysplasia',
                       'Hypomyelination with congenital cataract	; ORPHA:85163': 'Hypomyelination',
                       'NKX6-2-related hypomyelination	; ORPHA:527497': 'NKX6-2',
                       'Hikeshi-related hypomyelination	; ORPHA:495844': 'Hikeshi',
                       'Waardenburg syndrome type 2E	; ORPHA:895': 'Waardenburg',
                       'Fatty Acid Hydroxylase-associated neurodegeneration/ Spastische Spinalparalyse Typ 35 ('
                       'SPG35)	; ORPHA:171629': 'SPG35',
                       'Hereditary spastic paraplegia 11': 'spastic paraplegia 11',
                       'Hypomyelinating Leukodystrophy type 15 (ERPS1)': 'ERPS1',
                       'Hypomyelinating leukodystrophy type 15 (ERPS1)': 'ERPS1',
                       'Ataxia-pancytopenia syndrome': 'Ataxia-Pancytopenia',
                       'other': 'other',
                       'Notch1-associated Leukodystrophy': 'NOTCH1',
                       'Multiple Sklerose': 'MS'}

transformation_dict_label_short = {'HTRA1-related autosomal dominant cerebral small vessel disease': 'HTRA1',
                                   'X-linked Adrenoleukodystrophy; ORPHA:139399': 'X-ALD',
                                   'Peroxisome biogenesis disorder ; ORPHA:79189': 'Peroxisome biogenesis disorder',
                                   'Zellweger syndrome	; ORPHA:912': 'Zellweger syndrome',
                                   'Morbus Refsum	; ORPHA:772': 'Morbus Refsum',
                                   'α-Methylacyl-CoA racemase deficiency	; ORPHA:79095': 'α-Methylacyl-CoA',
                                   'Metachromatic leukodystrophy	; ORPHA:512': 'Meachromatic LD',
                                   'Krabbe disease	; ORPHA:487': 'Krabbe disease',
                                   'Multiple Sulfatase Deficiency	; ORPHA:585': 'Multiple Sulfatase',
                                   'Salla disease	; ORPHA:309334': 'Salla disease',
                                   'Fucosidosis	; ORPHA:349': 'Fucosidosis',
                                   'VPS11-related hypomyelination	; ORPHA:466934': 'VPS11',
                                   'Kearns Sayre syndrome (KSS)	; ORPHA:480': 'KSS',
                                   'Complex I defects	; ORPHA: 2609 u.a.': 'Complex I',
                                   'Complex II defects	; ORPHA: 3208': 'Complex II',
                                   'Complex III defects	; ORPHA: 1460': 'Complex III',
                                   'Complex IV defects	; ORPHA: 254905': 'Complex IV',
                                   'Iron sulfur cluster defects	; ORPHA:457406': 'Fe sulfur cluster',
                                   'Pyruvate carboxylase deficiency	; ORPHA:3008': 'Pyruvate carboxylase',
                                   'Mitochondrial neuro-gastrointestinal encephalopathy (MNGIE)	; ORPHA:298': 'MNGIE',
                                   'Canavan disease	; ORPHA:141': 'Canavan disease',
                                   'L-2-hydroxyglutaric aciduria	; ORPHA:79314': 'L-2-hydroxyglutaric aciduria',
                                   'Maple syrup urine disease (MSUD)	; ORPHA:268145': 'MSUD',
                                   'Phenylketonuria (PKU)	; ORPHA:716': 'PKU',
                                   '3-Hydroxy-3-methylglutaryl-CoA (HMGCoA) lyase deficiency	; ORPHA:35701': 'HMGCoA',
                                   'Defects in cobalamin, homocysteine and folate metabolism	; ORPHA:79282': 'cobalamin,..., meta',
                                   'Cerebral folate transport deficiency	; ORPHA:217382': 'Cerebral folate',
                                   'GABA transaminase deficiency	; ORPHA:2066': 'GABA',
                                   'Vanishing white matter (VWM)	; ORPHA:135': 'VWM',
                                   '4H syndrome	; ORPHA:289494': '4H syndrome',
                                   'Leukodystrophy with brainstem and spinal cord involvement and lactate elevation (LBSL)	; '
                                   'ORPHA:137898': 'LBSL',
                                   'Leukodystrophy with thalamus and brainstem cord involvement and lactatem elevation (LTBL)	; '
                                   'ORPHA:314051': 'LTBL',
                                   'AARS2-related leukodystrophy	; ORPHA:313808': 'AARS2',
                                   'Hypomyelination with brainstem and spinal cord involvement and leg spasticity (HBSL)	; '
                                   'ORPHA:363412': 'HBSL',
                                   'RARS-related hypomyelination	; ORPHA:438114': 'RARS',
                                   'EPRS-related hypomyelination': 'EPRS',
                                   'AIMP1-related hypomyelination	; ORPHA:280293': 'AIMP1',
                                   'Megalencephalic leukodystrophy with subcortical cysts (MLC)	; ORPHA:2478': 'MLC',
                                   'CLCN2-related leukodystrophy	; ORPHA:363540': 'CLCN2',
                                   'X-linked Charcot-Marie-Tooth disease (CMTX)	; ORPHA:64747': 'CMTX',
                                   'Congenital muscular dystrophies': 'Congenital muscular dystrophies',
                                   'GPR56-related leukodystrophy': 'GPR56',
                                   'MYRF-related leukodystrophy': 'MYRF',
                                   'Pelizaeus-Merzbacher disease (PMD)	; ORPHA:702': 'PMD',
                                   'Pelizaeus-Merzbacher-like disease	; ORPHA:280270': 'PMD like',
                                   'Oligodentodigital dysplasia (ODDD)	; ORPHA:2710': 'ODDD',
                                   'Oculodentodigital dysplasia (ODDD)\t; ORPHA:2710': 'ODDD',
                                   'Cerebral autosomal dominant arteriopathy met subcortical infarcts and leukodystrophy ('
                                   'CADASIL)	; ORPHA:136': 'CADASIL',
                                   'Cerebral autosomal recessive arteriopathy met subcortical infarcts and leukodystrophy ('
                                   'CARASIL)	; ORPHA:199354': 'CARASIL',
                                   'Cathepsin A-related arteriopathy with strokes and leukodystrophy (CARASAL)	; ORPHA:575553':
                                       'CARASAL',
                                   'Hereditary cerebral amyloid angiopathy	; ORPHA:439254': 'cerebral amyloid angiopathy',
                                   'Fabry disease	; ORPHA:324': 'Fabry disease',
                                   'COL4A1-/COL4A2-related disease	; ORPHA:477759': 'COL4A1/2',
                                   'Retinal vasculopathy and cerebral leukodystrophy (RVCL)	; ORPHA:247691': 'RVCL',
                                   'Aicardi-Goutières syndrome	; ORPHA:51': 'Aicardi-Goutières',
                                   'RNASET2-related leukodystrophy	; ORPHA:85136': 'RNASET2',
                                   'Leukodystrophy with calcifications and cysts (LCC)	; ORPHA:542310': 'LCC',
                                   'Coates Plus	; ORPHA:313838': 'Coates Plus',
                                   'Cockayne syndrome	; ORPHA:191': 'Cockayne syndrome',
                                   'Trichothiodystrophy with hypomyelination	; ORPHA:33364': 'Trichothiodystrophy',
                                   'Alexander disease	; ORPHA:58': 'Alexander disease',
                                   'adult polyglucosan body disease	; ORPHA:206583': 'adult polyglucosan body',
                                   'Giant axonal neuropathy	; ORPHA:643': 'Giant axonal neuropathy',
                                   'Porphyria-related leukodystrophy	; ORPHA: 79276': 'Porphyria',
                                   'Cerebrotendinous xanthomatosis	; ORPHA:909': 'Cerebrotendinous xanthomatosis',
                                   'Incontinentia pigmenti (IP)	; ORPHA:464': 'IP',
                                   'Sjögren-Larssen syndrome	; ORPHA:816': 'Sjögren-Larssen syndrome',
                                   'Lowe oculocerebrorenal syndrome	; ORPHA:534': 'Lowe',
                                   'Fragile X tremor/ataxia syndrome	; ORPHA:93256': 'X tremor/ataxia',
                                   'Adult onset dominant leukodystrophy (ADLD)	; ORPHA:99027': 'ADLD',
                                   'Adult onset leukodystrophy with axonal spheroids and pigmented glia (ALSP) due to CSF1 R '
                                   'mutation; ORPHA:313808': 'ALSP-CSF1',
                                   'Nasu Hakola disease; ORPHA:2770': 'Nasu Hakola',
                                   'Dentatorubropallidoluysian atrophy (DRPLA)	; ORPHA:101': 'DRPLA',
                                   'Hypomyelination with atrophy of the basal ganglia and cerebellum (HABC) / TUBB4A-related '
                                   'hypomyelination	; ORPHA:139441': 'HABC',
                                   'TMEM106B-related hypomyelination': 'THRM106B',
                                   'X-linked hypomyelination with spondylometaphyseal dysplasia	; ORPHA:83629':
                                       'spondylometaphyseal dysplasia',
                                   'Hypomyelination with congenital cataract	; ORPHA:85163': 'Hypomyelination',
                                   'NKX6-2-related hypomyelination	; ORPHA:527497': 'NKX6-2',
                                   'Hikeshi-related hypomyelination	; ORPHA:495844': 'Hikeshi',
                                   'Waardenburg syndrome type 2E	; ORPHA:895': 'Waardenburg',
                                   'Fatty Acid Hydroxylase-associated neurodegeneration/ Spastische Spinalparalyse Typ 35 ('
                                   'SPG35)	; ORPHA:171629': 'SPG35',
                                   'Hereditary spastic paraplegia 11': 'spastic paraplegia 11',
                                   'Hypomyelinating Leukodystrophy type 15 (ERPS1)': 'ERPS1',
                                   'Hypomyelinating leukodystrophy type 15 (ERPS1)': 'ERPS1',
                                   'Ataxia-pancytopenia syndrome': 'Ataxia-Pancytopenia',
                                   'other': 'other',
                                   'Notch1-associated Leukodystrophy': 'NOTCH1',
                                   'Multiple Sklerose': 'MS'}

transformation_dict_raw_label = {1: 'HTRA1-related autosomal dominant cerebral small vessel disease',
                                 2: 'X-linked Adrenoleukodystrophy; ORPHA:139399',
                                 3: 'Peroxisome biogenesis disorder ; ORPHA:79189',
                                 4: 'Zellweger syndrome	; ORPHA:912',
                                 5: 'Morbus Refsum	; ORPHA:772',
                                 6: 'α-Methylacyl-CoA racemase deficiency	; ORPHA:79095',
                                 7: 'Metachromatic leukodystrophy	; ORPHA:512',
                                 8: 'Krabbe disease	; ORPHA:487',
                                 9: 'Multiple Sulfatase Deficiency	; ORPHA:585',
                                 10: 'Salla disease	; ORPHA:309334',
                                 11: 'Fucosidosis	; ORPHA:349',
                                 12: 'VPS11-related hypomyelination	; ORPHA:466934',
                                 13: 'Kearns Sayre syndrome (KSS)	; ORPHA:480',
                                 14: 'Complex I defects	; ORPHA: 2609 u.a.',
                                 15: 'Complex II defects	; ORPHA: 3208',
                                 16: 'Complex III defects	; ORPHA: 1460',
                                 17: 'Complex IV defects	; ORPHA: 254905',
                                 18: 'Iron sulfur cluster defects	; ORPHA:457406',
                                 19: 'Pyruvate carboxylase deficiency	; ORPHA:3008',
                                 20: 'Mitochondrial neuro-gastrointestinal encephalopathy (MNGIE)	; ORPHA:298',
                                 21: 'Canavan disease	; ORPHA:141',
                                 22: 'L-2-hydroxyglutaric aciduria	; ORPHA:79314',
                                 23: 'Maple syrup urine disease (MSUD)	; ORPHA:268145',
                                 24: 'Phenylketonuria (PKU)	; ORPHA:716',
                                 25: '3-Hydroxy-3-methylglutaryl-CoA (HMGCoA) lyase deficiency	; ORPHA:35701',
                                 26: 'Defects in cobalamin, homocysteine and folate metabolism	; ORPHA:79282',
                                 27: 'Cerebral folate transport deficiency	; ORPHA:217382',
                                 28: 'GABA transaminase deficiency	; ORPHA:2066',
                                 29: 'Vanishing white matter (VWM)	; ORPHA:135',
                                 30: '4H syndrome	; ORPHA:289494',
                                 31: 'Leukodystrophy with brainstem and spinal cord involvement and lactate elevation (LBSL)	; ORPHA:137898',
                                 32: 'Leukodystrophy with thalamus and brainstem cord involvement and lactatem elevation (LTBL)	; ORPHA:314051',
                                 33: 'AARS2-related leukodystrophy	; ORPHA:313808',
                                 34: 'Hypomyelination with brainstem and spinal cord involvement and leg spasticity (HBSL)	; ORPHA:363412',
                                 35: 'RARS-related hypomyelination	; ORPHA:438114',
                                 36: 'EPRS-related hypomyelination',
                                 37: 'AIMP1-related hypomyelination	; ORPHA:280293',
                                 38: 'Megalencephalic leukodystrophy with subcortical cysts (MLC)	; ORPHA:2478',
                                 39: 'CLCN2-related leukodystrophy	; ORPHA:363540',
                                 40: 'X-linked Charcot-Marie-Tooth disease (CMTX)	; ORPHA:64747',
                                 41: 'Congenital muscular dystrophies',
                                 42: 'GPR56-related leukodystrophy',
                                 43: 'MYRF-related leukodystrophy',
                                 44: 'Pelizaeus-Merzbacher disease (PMD)	; ORPHA:702',
                                 45: 'Pelizaeus-Merzbacher-like disease	; ORPHA:280270',
                                 46: 'Oculodentodigital dysplasia (ODDD)\t; ORPHA:2710',
                                 47: 'Cerebral autosomal dominant arteriopathy met subcortical infarcts and leukodystrophy (CADASIL)	; ORPHA:136',
                                 48: 'Cerebral autosomal recessive arteriopathy met subcortical infarcts and leukodystrophy (CARASIL)	; ORPHA:199354',
                                 49: 'Cathepsin A-related arteriopathy with strokes and leukodystrophy (CARASAL)	; ORPHA:575553',
                                 50: 'Hereditary cerebral amyloid angiopathy	; ORPHA:439254',
                                 51: 'Fabry disease	; ORPHA:324',
                                 52: 'COL4A1-/COL4A2-related disease	; ORPHA:477759',
                                 53: 'Retinal vasculopathy and cerebral leukodystrophy (RVCL)	; ORPHA:247691',
                                 54: 'Aicardi-Goutières syndrome	; ORPHA:51',
                                 55: 'RNASET2-related leukodystrophy	; ORPHA:85136',
                                 56: 'Leukodystrophy with calcifications and cysts (LCC)	; ORPHA:542310',
                                 57: 'Coates Plus	; ORPHA:313838',
                                 58: 'Cockayne syndrome	; ORPHA:191',
                                 59: 'Trichothiodystrophy with hypomyelination	; ORPHA:33364',
                                 60: 'Alexander disease	; ORPHA:58',
                                 61: 'adult polyglucosan body disease	; ORPHA:206583',
                                 62: 'Giant axonal neuropathy	; ORPHA:643',
                                 63: 'Porphyria-related leukodystrophy	; ORPHA: 79276',
                                 64: 'Cerebrotendinous xanthomatosis	; ORPHA:909',
                                 65: 'Incontinentia pigmenti (IP)	; ORPHA:464',
                                 66: 'Sjögren-Larssen syndrome	; ORPHA:816',
                                 67: 'Lowe oculocerebrorenal syndrome	; ORPHA:534',
                                 68: 'Fragile X tremor/ataxia syndrome	; ORPHA:93256',
                                 69: 'Adult onset dominant leukodystrophy (ADLD)	; ORPHA:99027',
                                 70: 'Adult onset leukodystrophy with axonal spheroids and pigmented glia (ALSP) due to CSF1 R mutation; ORPHA:313808',
                                 71: 'Nasu Hakola disease; ORPHA:2770',
                                 72: 'Dentatorubropallidoluysian atrophy (DRPLA)	; ORPHA:101',
                                 73: 'Hypomyelination with atrophy of the basal ganglia and cerebellum (HABC) / TUBB4A-related hypomyelination	; ORPHA:139441',
                                 74: 'TMEM106B-related hypomyelination',
                                 75: 'X-linked hypomyelination with spondylometaphyseal dysplasia	; ORPHA:83629',
                                 76: 'Hypomyelination with congenital cataract	; ORPHA:85163',
                                 77: 'NKX6-2-related hypomyelination	; ORPHA:527497',
                                 78: 'Hikeshi-related hypomyelination	; ORPHA:495844',
                                 79: 'Waardenburg syndrome type 2E	; ORPHA:895',
                                 80: 'Fatty Acid Hydroxylase-associated neurodegeneration/ Spastische Spinalparalyse Typ 35 (SPG35)	; ORPHA:171629',
                                 81: 'Cerebral Amyloid Angiopathy (CAA)',
                                 82: 'Antiphospholipidsyndrom',
                                 83: 'PRES',
                                 84: 'Multiple Sklerose',
                                 85: 'Neuromyelitis optica (Devic Krankheit)',
                                 86: 'Zentrale pontine Myelinolyse',
                                 87: 'ADEM',
                                 88: 'Progressive multifokale Enzephalopathie',
                                 89: 'HIV Enzephalitis',
                                 90: 'CMV Enzephalitis',
                                 91: 'SSPE',
                                 92: 'Gliomatosis cerebri',
                                 93: 'ZNS Lymphome',
                                 94: 'Vaskuläre Enzephalopathie',
                                 95: 'Churg-Strauss Vaskulitis',
                                 96: 'Mb Behcet',
                                 97: 'Systemischer Lupus',
                                 98: 'Systemische Sklerodermie',
                                 99: 'Morbus Sjögren',
                                 100: 'Neurosarkoidose',
                                 101: 'Neuroborreliose',
                                 102: 'preterm brain injury (PVL)',
                                 103: 'Hereditary spastic paraplegia 11',
                                 104: 'Hypomyelinating leukodystrophy type 15 (ERPS1)',
                                 105: 'Ataxia-pancytopenia syndrome'
                                 }



def get_instruments_from_redcap_data(redcap_data: pd.DataFrame, instruments: list, old_version: bool) -> pd.DataFrame:
    """
    Function for extracting the form you want
    :param redcap_data: data from the redcap system with the specific structure
    :param instruments: a list of string which you like to extract
    :param old_version: if baseline is not coded
    :return: dataframe with selected form
    """
    redcap_instruments = redcap_data['redcap_repeat_instrument']
    if old_version:
        redcap_instruments = redcap_instruments.replace(np.nan, 'Baseline')
    redcap_data['redcap_repeat_instrument'] = redcap_instruments
    redcap_data_selected = redcap_data[redcap_data['redcap_repeat_instrument'].isin(instruments)]
    return redcap_data_selected


def try_parsing_date(text):
    for fmt in ('%Y-%m-%d', '%m-%Y', '%y', '%Y-%m-%d', '%m.%Y', '%m/%y', '%m/%Y', '%Y', '%Y.%w', '00-%Y'):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    raise ValueError('no valid date format found')


def convert_series_to_datetimes(colname: str, data: pd.DataFrame):
    dates = data[colname]
    dates = dates.dropna()
    dates = dates.astype(str)
    records = dates.index
    dates_list = dates.to_list()
    dates_list_format = []
    for date in dates_list:
        dates_list_format.append(try_parsing_date(date))
    df = pd.DataFrame(list(zip(records.to_list(), dates_list_format)), columns=['record_id', colname])
    return df


def save_hist_plot(output_path: str, parameter: list, label_x: str) -> None:
    """

    :param output_path: path to save the images
    :param parameter:
    :param label_x:
    """
    # save histogram as csv
    hist = np.histogram(parameter, bins=k_anyme_bins(parameter,5))
    hist_df = pd.DataFrame(list(zip(hist[1][:-1], hist[1][1:], hist[0])), columns=['left', 'right', 'count'])
    hist_df.to_csv(output_path.replace('.png', '.csv'), index=False)
    plt.figure(0)
    plt.rcParams.update({'figure.figsize': (7, 6), 'figure.dpi': 100})
    plt.hist(parameter, bins=k_anyme_bins(parameter,5), edgecolor='black', linewidth=1.2)

    plt.gca().set(title='Frequency of {} (n={})'.format(label_x, len(parameter)), ylabel='Frequency',
                  xlabel=label_x)
    plt.subplots_adjust(top=0.925,
                        bottom=0.2,
                        left=0.15,
                        right=0.90,
                        hspace=0.01,
                        wspace=0.01)
    plt.savefig(output_path)


def k_anyme_bins(parameters: list, k: int = 5) -> list:
    """
    chrates a list of bins for the histogram that has a step size of k elements from parameters
    :param parameters: list of values
    :param k: minimum number of elements in a bin
    :return: list of bins
    """
    # sort the parameters
    parameters = sorted(parameters)

    # throw exception if parameter have less than k elements
    if len(parameters) < k+2:
        raise ValueError('parameter has less than k elements')

    # find the first binwidth that allows to have at least k elements in the bin
    for i in reversed(range(1, max(parameters) - min(parameters))):
        # split the list into bins
        hist, bins = np.histogram(parameters, bins=i)
        hist = [i for i in hist if i != 0]
        # check if all bins have at least k elements
        if all( x >= k + 1 for x in hist):
            break
    return bins

def k_anonym_count(name_list: list,count_list: list, k: int = 5) -> int:
    """
    gets a list of names and a list of the same length with the count of the names,and a k value
    returns the name_list and count list with the names that have less than k elements merged into the other category
    :param name_list: list of names
    """
    #TODO remove print statements
    #print(name_list)
    #print(count_list)
    #get index of names that have less than k elements
    index = [i for i, x in enumerate(count_list) if x <= k]
    #get the sum of all elements that have less than k elements
    sum = 0
    for i in index:
        sum += count_list[i]
    #add the sum to a new element in the count list
    count_list.append(sum)
    #add the name of the new element as the combined name of all elements that have less than k elements
    name = ''
    for i in index:
        name += name_list[i] + ', '
    name_list.append(name[:-2])
    #delete all elements that have less than k elements
    for i in reversed(index):
        del count_list[i]
        del name_list[i]

    return name_list ,count_list

def export_to_redcap_via_pycap(api_url: str, api_key: str) -> pd.DataFrame:
    """
    :param api_url: URL to the REDCAP API as String
    :param api_key: API Key for the Project
    :return: pandas dataframe of all records
    """
    project = Project(api_url, api_key)
    df = project.export_records(format_type="df", raw_or_label="raw")
    return df


def export_metadata_via_pycap(api_url: str, api_key: str) -> pd.DataFrame:
    """
    :param api_url: URL to the REDCAP API as String
    :param api_key: API Key for the Project
    :return: pandas dataframe of the metadata
    """
    project = Project(api_url, api_key)
    df = project.export_metadata(format_type="df")
    return df


def export_to_redcap_via_request(api_url: str, api_key: str) -> str:
    """

    """
    data = {
        'token': api_key,
        'content': 'record',
        'action': 'export',
        'format': 'csv',
        'type': 'flat',
        'csvDelimiter': '',
        'records[0]': '2',
        'rawOrLabel': 'label',
        'rawOrLabelHeaders': 'raw',
        'exportCheckboxLabel': 'false',
        'exportSurveyFields': 'false',
        'exportDataAccessGroups': 'false',
        'returnFormat': 'json'
    }
    r = requests.post(api_url, data=data)
    df = r.text
    return df


def load_redcap_metadata_from_file(file_path: str) -> pd.DataFrame:
    """
     a simple function for reading the redcap files
    @type file_path: path to the redcap file
    """
    df = pd.read_csv(file_path, index_col=0)
    return df


def load_redcap_data_from_file(file_path: str) -> pd.DataFrame:
    """
     a simple function for reading the redcap files
    @type file_path: path to the redcap file
    """
    df = pd.read_csv(file_path, index_col=[0, 1])
    return df


def print_pdf_line(table: pd.Series, pdf: FPDF) -> None:
    """

    :param table: results of a value_count of a series
    :param pdf: pdf object where you print it in
    """
    for table_row in range(0, len(table)):
        value = str(table[table_row])
        name = str(table.index[table_row])
        pdf.cell(100, 8, txt="{} \t {}".format(name, value), ln=2, align='L')


def print_table_to_pdf(table: pd.Series, pdf: FPDF, header_included: bool, header: list) -> None:
    """

    :param header_included: is the header included in the dataframe
    :param header: a list of header (number of columns +1)
    :param table: results of a value_count of a series
    :param pdf: pdf object where you print it in
    """
    epw = pdf.w - 2 * pdf.l_margin
    line_height = pdf.font_size * 2.5
    col_width = epw / 2
    if not header_included:
        for head in header:
            pdf.cell(col_width, line_height, head, border=1, align='C')
        pdf.ln(line_height)
    for table_row in range(0, len(table)):
        value = str(table[table_row])
        name = str(table.index[table_row])
        pdf.cell(col_width, line_height, name, border=1, align='C')
        pdf.cell(col_width, line_height, value, border=1, align='C')
        pdf.ln(line_height)


def add_image_to_pdf(pdf, name, x, y, h, w):
    if os.path.exists(name):
        pdf.image(name=name, x=x, y=y, w=w, h=h)


def create_pdf(station_name, number_of_patients, number_of_mris, number_of_examinations, number_of_genetic, table_sex,
               table_examination,
               number_of_diagnosis, table_diagnosis, overview_plot_path, overview_accumulated_plot_path, hist_plot_path,
               exam_plot_path, mri_plot_path,
               genetic_plot_path, output_path):
    # save FPDF() class into a
    pdf = FPDF()
    # start the first page
    pdf.add_page()
    # set style for the Title
    pdf.set_font("Arial", size=24, style='BU')
    # create a cell
    pdf.cell(200, 25, txt="Train Report {}".format(station_name), ln=1, align='C')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(180, 5, border=0, align='L', txt="This report is for the automatic generation of an overview of the "
                                                    "structured RedCAP data at the respective locations. For the "
                                                    "report, the 4 different masks Baseline, Examination, MRI and "
                                                    "Genetics are extracted and an overview is generated. If you have"
                                                    " any questions, suggestions or criticism, please contact "
                                                    "Lars Hempel.")
    # add another cell
    pdf.set_font("Arial", size=16, style="U")
    pdf.line(10, 60, 200, 60)
    pdf.cell(100, 20, txt="Overview:", ln=2, align='L')
    pdf.set_font("Arial", size=15)
    pdf.multi_cell(180, 7, border=0, align='L', txt="The overview is done on all questionnaires of the patients. A "
                                                    "baseline questionnaire is created for each patient and the "
                                                    "questionnaires for examination, MRI or genetics can be filled "
                                                    "in. MRI and Examination are longitudinale therefeore multiple filled out."
                                                    " The next graph shows the combination of all questionnaires.")
    pdf.cell(100, 8, txt="Number of Patients: {}".format(number_of_patients), ln=2, align='L')
    add_image_to_pdf(pdf=pdf, name=overview_accumulated_plot_path, x=10, y=125, w=200, h=200)
    pdf.add_page()
    pdf.multi_cell(180, 7, border=0, align='L', txt="In this plot, all combinations of the different questionnaires "
                                                    "are displayed. It is only counted whether a questionnaire is "
                                                    "present and not how many are present. Thus, the sum of all "
                                                    "baseline questionnaires equals the number of patients and the "
                                                    "sum of the MRI questionnaires equals the total number of "
                                                    "patients with at least one MRI questionnaire.")
    add_image_to_pdf(pdf, name=overview_plot_path, x=10, y=80, w=200, h=200)
    pdf.add_page()

    # create chapter examination
    pdf.set_font("Arial", size=20, style="U")
    pdf.cell(100, 20, txt="Chapter: Examination", ln=1, align='L')
    pdf.set_font("Arial", size=15)
    pdf.multi_cell(180, 7, border=0, align='L', txt="The examination questionnaire contains the phenotypic information "
                                                    "about the patient and can be filled out at each visit. Therefore "
                                                    "representing longitudinal information. Only the number of "
                                                    "follow-ups and the number of questionnaires per patient are "
                                                    "examined.")
    pdf.set_font("Arial", size=15)
    pdf.cell(100, 15, txt="Number of Examination questionnaires: {}".format(number_of_mris), ln=2, align='L')
    pdf.set_font("Arial", size=16, style="U")
    pdf.cell(100, 15, txt="Follow Up", ln=1, align='L')
    pdf.set_font("Arial", size=15)
    add_image_to_pdf(pdf, exam_plot_path, x=0, y=140, w=200, h=150)
    pdf.add_page()

    # create chapter Baseline
    pdf.set_font("Arial", size=20, style="U")
    pdf.cell(100, 20, txt="Chapter: Baseline", ln=1, align='L')
    pdf.set_font("Arial", size=15)
    pdf.multi_cell(180, 7, border=0, align='L', txt="The baseline questionnaires contain basic data about the "
                                                    "patient. The categories examined are gender, age at diagnosis "
                                                    "and the diagnoses represented.")
    pdf.set_font("Arial", size=16, style="U")
    pdf.cell(100, 15, txt="Sex", ln=1, align='L')
    pdf.set_font("Arial", size=15)
    print_table_to_pdf(table=table_sex, pdf=pdf, header_included=False, header=["Sex", "Number"])
    pdf.set_font("Arial", size=16, style="U")
    pdf.cell(100, 15, txt="Age", ln=1, align='L')
    add_image_to_pdf(pdf, name=hist_plot_path, x=0, y=120, w=200, h=150)
    pdf.add_page()
    pdf.set_font("Arial", size=16, style="U")
    pdf.cell(100, 15, txt="Diagnoses", ln=1, align='L')
    pdf.set_font("Arial", size=15)
    pdf.cell(100, 20, txt="Number of Diagnoses: {}".format(number_of_diagnosis), ln=2, align='L')
    #print_table_to_pdf(table=table_diagnosis, pdf=pdf, header_included=False,
    #                   header=["Name Diagnoses", "Number of Diagnoses"])
    pdf.add_page()

    # create chapter MRI
    pdf.set_font("Arial", size=20, style="U")
    pdf.cell(100, 20, txt="Chapter: MRI", ln=1, align='L')
    pdf.set_font("Arial", size=15)
    pdf.multi_cell(180, 7, border=0, align='L', txt="The MRI questionnaire contains information on the patients "
                                                    "radiological data image processing about the patient and can be "
                                                    "filled out t every visit The data points to the respective files "
                                                    "in the database for the MRI images. Only the number of follow "
                                                    "ups, i.e. the number of questionnaires per patient, is examined.")
    pdf.set_font("Arial", size=15)
    pdf.cell(100, 15, txt="Number of MRIs questionnaires: {}".format(number_of_mris), ln=2, align='L')
    pdf.set_font("Arial", size=16, style="U")
    pdf.cell(100, 15, txt="Follow Up", ln=1, align='L')
    add_image_to_pdf(pdf, name=mri_plot_path, x=0, y=90, w=200, h=150)
    pdf.add_page()

    # create chapter genetics
    pdf.set_font("Arial", size=20, style="U")
    pdf.cell(100, 20, txt="Chapter: Genetics", ln=1, align='L')
    pdf.set_font("Arial", size=15)
    pdf.multi_cell(180, 7, border=0, align='L', txt="The genetic questionnaire contains the genetic characteristics "
                                                    "of a patient and are only filled out once. The data indicate "
                                                    "provides information about the classification and used "
                                                    "diagnostics. In this chapter, we describe the genetic findings "
                                                    "listed.")
    pdf.set_font("Arial", size=15)
    pdf.cell(100, 15, txt="Number of Genetic questionnaires: {}".format(number_of_genetic), ln=2, align='L')
    pdf.set_font("Arial", size=16, style="U")
    pdf.cell(100, 15, txt="Findings", ln=1, align='L')
    add_image_to_pdf(pdf, name=genetic_plot_path, x=0, y=90, w=200, h=150)
    # save the pdf
    pdf.output(output_path)


def get_instrument_df(redcap_data: pd.DataFrame, redcap_metadata: pd.DataFrame, instrument: str, with_complete: bool =
True, station: str = None) -> pd.DataFrame:
    """
    This function should extract the instruments from the redcap structured dataframe
    :param with_complete: a boolean if the complete_instrument column should be included
    :param instrument: a string with the instrument name in it
    :param redcap_metadata: dataframe of the redcap system
    :type redcap_data: dataframe of the redcap system
    """
    # get the specific metadata for the instrument
    metadata_instrument = redcap_metadata[redcap_metadata['form_name'] == instrument]
    if instrument is "examination_data" and metadata_instrument.empty:
        metadata_instrument = redcap_metadata[
            redcap_metadata['form_name'] == "examination_data_use_new_sheet_for_every_visit"]
        instrument = "examination_data"
    # declare the start and stop point
    if instrument == 'basic_data_consent':
        start_field_name = metadata_instrument.index[1]
    else:
        start_field_name = metadata_instrument.index[0]

    if instrument != 'genetics':
        redcap_event_name = instrument
    else:
        redcap_event_name = "basic_data_consent"

    if instrument == 'mri' and station is "Tuebingen":
        end_field_name = metadata_instrument.index[-2]
        start_field_name = metadata_instrument.index[1]
    elif instrument == 'mri' and station is not "Tuebingen":
        end_field_name = metadata_instrument.index[-2]
    else:
        end_field_name = metadata_instrument.index[-1]
    rows = redcap_data["redcap_repeat_instrument"] == redcap_event_name
    # add 1 to the end position to cover the complete_instrument column
    end_field_number = np.where(redcap_data.columns == end_field_name)
    if with_complete:
        end_field_name = redcap_data.columns[end_field_number[0][0] + 1]
    else:
        end_field_name = redcap_data.columns[end_field_number[0][0]]
    # extract the data
    instrument_df = redcap_data.loc[rows, start_field_name:end_field_name]
    complete_col = redcap_data.loc[rows, end_field_name]
    instrument_df = instrument_df.loc[complete_col.notna(), :]
    return instrument_df


def get_record_id_to_instruments(redcap_data: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """
    This function should extract the instruments from the redcap structured dataframe
    :param with_complete: a boolean if the complete_instrument column should be included
    :param instrument: a string with the instrument name in it
    :param redcap_metadata: dataframe of the redcap system
    :type redcap_data: dataframe of the redcap system
    """
    if instrument != 'genetics':
        redcap_event_name = instrument
    else:
        redcap_event_name = "basic_data_consent"
    rows = redcap_data["redcap_repeat_instrument"] == redcap_event_name
    # extract the data
    instrument_df = redcap_data.loc[rows, 'record_id']

    return instrument_df


def get_transformation_dict_from_answers(metadata: pd.DataFrame, field_name: str) -> dict:
    """
    @param field_name: str with responding column name
    @type metadata: dataframe form redcap about meta information
    """
    meta_field = metadata[metadata.index == field_name]
    answers = meta_field['select_choices_or_calculations']
    answers_split_list = answers.values[0].split("|")
    answers_numbers = [int(i.split(',')[0]) for i in answers_split_list]
    answers_character = [i.split(',')[1] for i in answers_split_list]
    map_dict = dict(zip(answers_numbers, answers_character))
    return map_dict

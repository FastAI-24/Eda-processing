
"""
Step 5-1: 범주형 변수 결측치 진단
데이터셋 내의 범주형(Object) 컬럼들의 결측치 현황을 정밀 분석합니다.
단순히 NaN(Not a Number)만 찾는 것이 아니라, 데이터 수집 과정에서 발생할 수 있는
텍스트 형태의 결측 표기('-', 'NULL', ' ', '0' 등)도 함께 식별합니다.

이 정보는 다음 단계인 '범주형 데이터 정제(Cleanup)'의 기초 자료로 활용됩니다.
"""

import pandas as pd
import numpy as np

def check_categorical_missing_rate():
    print("[Step 3-2] Checking Missing Rate of Categorical Features...")
    
    # 데이터 로드 (Step 3 산출물)
    data_path = '../../data/analysis_steps/step3_feature_engineered.csv'
    df = pd.read_csv(data_path)
    
    # Object 타입(문자열/범주형) 컬럼만 선택
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 분석에서 제외할 명백한 식별자/주소 컬럼들
    exclude = ['시군구', '아파트명', '도로명', '구', '동', '지번주소', '계약년월', '계약일자']
    target_cats = [c for c in cat_cols if c not in exclude]
    
    results = []
    
    for col in target_cats:
        # 1. NaN 결측치 비율 계산
        n_missing = df[col].isnull().sum()
        ratio = (n_missing / len(df)) * 100
        n_unique = df[col].nunique()
        
        # 2. 최빈값(Mode) 확인 (가장 흔한 값이 무엇인지 파악)
        top_val = df[col].mode()[0] if n_unique > 0 else "None"
        top_freq = (df[col].value_counts().iloc[0] / len(df) * 100) if n_unique > 0 else 0
        
        results.append({
            'Column': col,
            'Missing Rate (%)': ratio,
            'Unique Categories': n_unique,
            'Top Value': top_val,
            'Top Value Rate (%)': top_freq
        })
        
    # 결측률이 높은 순서대로 정렬하여 출력
    results.sort(key=lambda x: x['Missing Rate (%)'], reverse=True)
    
    # 리포트 파일 생성
    with open('categorical_missing_report.txt', 'w', encoding='utf-8') as f:
        f.write(f"Categorical Feature Missing Rate ({len(results)} columns)\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Column Name':<30} | {'Missing (%)':<12} | {'Unique':<6} | {'Top Value':<15} | {'Top Rate (%)'}\n")
        f.write("-" * 100 + "\n")
        
        for r in results:
            f.write(f"{r['Column']:<30} | {r['Missing Rate (%)']:<12.2f} | {r['Unique Categories']:<6} | {r['Top Value']:<15} | {r['Top Value Rate (%)']:.2f}\n")
            
        f.write("-" * 100 + "\n")
        f.write("Tip: If Missing rate > 70%, consider dropping unless strongly correlated.\n")
        
    print("Analysis Complete. Report saved to 'categorical_missing_report.txt'")

if __name__ == "__main__":
    check_categorical_missing_rate()

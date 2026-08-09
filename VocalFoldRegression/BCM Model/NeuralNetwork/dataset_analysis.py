import os
import pandas as pd

BCM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # VocalFoldRegression/BCM Model

female = pd.read_csv(os.path.join(BCM_DIR, 'FemaleBCM.csv'))
updated_female = female[female['ACFL'] > 30]
male = pd.read_csv(os.path.join(BCM_DIR, 'MaleBCM.csv'))

print(f"---F0 comparison---")
print(f"female_min {female['F0'].min()} | female_max {female['F0'].max()} | female_mean {female['F0'].mean()} | female_std {female['F0'].std()}")
print(f"male_min {male['F0'].min()} | male_max {male['F0'].max()} | male_mean {male['F0'].mean()} | male_std {male['F0'].std()}")

print(f"\n---SPL comparison---")
print(f"female_min {female['SPL'].min()} | female_max {female['SPL'].max()} | female_mean {female['SPL'].mean()} | female_std {female['SPL'].std()}")
print(f"male_min {male['SPL'].min()} | male_max {male['SPL'].max()} | male_mean {male['SPL'].mean()} | male_std {male['SPL'].std()}")

print(f"\n---Updated Female (ACFL > 30) - F0 comparison---")
print(f"updated_female_min {updated_female['F0'].min():.2f} | updated_female_max {updated_female['F0'].max():.2f} | updated_female_mean {updated_female['F0'].mean():.2f} | updated_female_std {updated_female['F0'].std():.2f}")
print(f"Coefficient of Variation: {(updated_female['F0'].std() / updated_female['F0'].mean() * 100):.1f}%")

print(f"\n---Updated Female (ACFL > 30) - SPL comparison---")
print(f"updated_female_min {updated_female['SPL'].min():.2f} | updated_female_max {updated_female['SPL'].max():.2f} | updated_female_mean {updated_female['SPL'].mean():.2f} | updated_female_std {updated_female['SPL'].std():.2f}")
print(f"Coefficient of Variation: {(updated_female['SPL'].std() / updated_female['SPL'].mean() * 100):.1f}%")

print(f"\n---Dataset Sizes---")
print(f"Original female: {len(female)} samples")
print(f"Updated female (ACFL > 30): {len(updated_female)} samples ({len(updated_female)/len(female)*100:.1f}% retained)")
print(f"Male: {len(male)} samples")

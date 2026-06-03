# Model Summary

## Задача

Бинарная классификация: предсказать вероятность `smoking=1` по медицинским и антропометрическим признакам.

## Данные

- Train: 15 000 строк, 23 исходных признака + target
- Test: 10 000 строк, 23 исходных признака
- Target balance: около 36.7% positive class
- После feature engineering: 43 признака

## Лучшие результаты из ноутбука

| Конфигурация | OOF ROC-AUC |
|---|---:|
| Expert weighted blend | 0.8889 |
| Uncertainty-gated blend | 0.8886 |
| Cleanlab remove top 3% / clean expert | 0.8885 |
| Full tree blend | 0.8879 |
| Tuned XGBoost | 0.8873 |

Public leaderboard score: 0.88012.

## Основные выводы

- SMOTE не дал прироста для бустингов и ухудшал часть конфигураций.
- Cleanlab помог найти потенциально шумные/сложные объекты и дал небольшой OOF-прирост.
- Proxy/interactions признаки на основе роста, гемоглобина, креатинина и BMI стали важными для CatBoost.
- Label-dependent gating был исключён из финального выбора, потому что завышал OOF и не переносился на test.

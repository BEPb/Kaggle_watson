### План работы

- EDA (exploratory data analysis). статистики по датасету, боксплоты, разброс категорий, ...
- Data Cleaning — все, что касается очистки данных. Выбросы, пропуски, и т.д. (пропуски через fillna, чистка 
  категорий, объединение категорий) 
- Data preparation — все, что касается подготовки данных для модели. Несколько блоков:
  * Общий (обработка категорий — label/ohe/frequency, проекция числовых на категории, трансформация числовых, бининг)
  * Для регрессий/нейронных сетей (различное масштабирование)
  * Для деревьев
  * Специальный (временные ряды, картинки, FM/FFM)
  * Текст (Vectorizers, TF-IDF, Embeddings)
- Models
  * Linear models (различные регрессии — ridge/logistic)
  * Tree models (lgb)
  * Neural Networks
  * Exotic (FM/FFM)
- Feature selection (grid/random search)
- Hyperparameters search
- Ensemble (Regression / lgb)

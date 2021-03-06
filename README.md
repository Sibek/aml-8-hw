# План проекта
## Предсказание оценки качества вина на основе данных из физико-химических свойств
### Введение
В данном анализе исследуется влияние физико-химических свойств вина на экспертную оценку качества.
В анализе рассматривается набор данных с красными и белыми португальскими винами "Vinho Verde". Доступны только физико-химические и сенсорные переменные (например, нет данных о типах винограда, марке вина, цене продажи вина и т.д.).

### Цель проводимого анализа
Основная цель - изучить свойства вина, которые наибольшим образом влияют на экспертную оценку качества и определить наиболее подходящий алгоритм для обучения модели, обучить модель предсказывать результаты.
### Рассматриваемые в анализе физико-химические свойств вина
Всего в датасете 12 физико-химических свойств вина, которые мы будем использовать для предсказания целевого показателя - оценки качества вина.
- type - тип вина (красное/белое)
- fixed acidity - фиксированная кислотность
- volatile acidity - летучая кислотность
- citric acid - лимонная кислота
- residual sugar - остаточный сахар
- chlorides - хлориды
- free sulfur dioxide - свободный диоксид серы - total sulfur dioxide - общий диоксид серы
- density - плотность
- pH
- sulphates - сульфаты
- alcohol - алкогольность

### Применяемые алгоритмы
- LinearRegression
- RandomForestClassifier
- KNeighborsClassifier
- XGBClassifier
- GradientBoostingClassifier
- DecisionTreeClassifier
- GaussianNB
- SVC

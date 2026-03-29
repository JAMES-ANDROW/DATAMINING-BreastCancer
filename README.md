# Breast Cancer Classifier (WEKA)

This program loads `breast-cancer.arff`, trains and evaluates all required models using **10-fold cross-validation**, and then starts an interactive prediction loop.

## Required Models Included

- ZeroR
- OneR
- IBk
- Naive Bayes
- J48
- Logistic Regression

## Prediction Model Choice

For the interactive prediction phase, the program first lets the user choose which trained model to use:

- ZeroR
- OneR
- IBk
- Naive Bayes
- J48
- Logistic Regression

Team default is **J48** if the user presses Enter or enters an invalid option.

This default choice is stated in source comments in `BABAD_BrestCancerClassifierApp.java` and here in the README.

## Requirements

- Java compiler/runtime compatibility (use matching versions, or compile for Java 8)
- WEKA library JAR (for example: `weka-stable-3.8.x.jar`)
- `breast-cancer.arff` in the same folder, or pass its path as an argument

## Compile

```bash
javac --release 8 -cp ".;weka.jar" BABAD_BrestCancerClassifierApp.java
```

(Replace `weka.jar` with your actual WEKA jar filename/path.)

If your `javac` and `java` are already the same modern version, you can omit `--release 8`.

## Run

```bash
java -cp ".;weka.jar" BABAD_BrestCancerClassifierApp
```

Or with a custom dataset path:

```bash
java -cp ".;weka.jar" BABAD_BrestCancerClassifierApp path\\to\\breast-cancer.arff
```

## Quick Run (Windows PowerShell)

```powershell
javac --release 8 -cp ".;weka-stable-3.8.x.jar" BABAD_BrestCancerClassifierApp.java
java -cp ".;weka-stable-3.8.x.jar" BABAD_BrestCancerClassifierApp
```

## Output Behavior

For each model, the program prints:

- model name,
- summary of results,
- accuracy (`Correctly Classified`),
- confusion matrix.

After evaluation, it enters an interactive loop:

1. Prompts for each non-class attribute.
2. Predicts class and displays the result.
3. Asks whether to predict again (`yes`/`y` to continue, `no`/`n` to stop).
4. Exits gracefully with a closing message.

/* Intializa all libraries and ofc ang weka dataset thru wekajar file */

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import java.util.Scanner;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class BreastCancerClassifierApp {

    /*
     * Team name: BABAD (BA+BAD)
     * Members: Florence Elaine Soleño, Duane Ryann Montes, James Andrew Agustin
     */
    
    // ===== REFLECTIVE NOTES =====
    // 
    // Model Selection (J48 Default):
    // The team chose J48 (decision tree) as the default prediction model because:
    // - It provides interpretable rules that clinicians can understand and validate.
    // - It balances accuracy (~75% cross-validation) with practical usability.
    // - Unlike black-box models, J48 outputs can be inspected to verify reasoning.
    //
    // Invalid Input Handling:
    // The team implemented multi-layered validation to reduce user frustration:
    // - Case-insensitive matching (e.g., "yes", "YES", "Yes" all map to attribute value).
    // - Range-based nominal attributes: accepts numeric input (e.g., user enters "45" 
    //   for age range "40-49") and auto-maps to the correct category.
    // - Missing value defaults: blank input silently sets attribute to missing, which
    //   WEKA handles gracefully in prediction using built-in statistical inference.
    // - Repeated prompts on invalid input avoid program crash and guide users to valid choices.
    //
    // Testing Observations:
    // - Initial strict input validation (exact text match only) caused excessive 
    //   re-prompting; relaxing to case-insensitive + numeric-range matching 
    //   improved user experience significantly.
    // - The interactive loop benefits from blank-line spacing between prompts and outputs;
    //   terminal readability was poor without it, so printPrompt() and printInfo() helpers
    //   were added to enforce consistent formatting.
    // - Cross-validation on this cleaned dataset reveals moderate class imbalance
    //   (201 no-recurrence vs 85 recurrence instances), which explains why 
    //   recall on the minority class remains a challenge even for the best model.
    //
    
    // Team default for interactive prediction model: J48 decision tree.
    // Pero Users can also choose a different model at runtime.
    private static final String DEFAULT_PREDICTION_MODEL_NAME = "J48";


    private static final String[] PREDICTION_MODEL_NAMES = new String[] {
        "ZeroR",
        "OneR",
        "IBk",
        "Naive Bayes",
        "J48",
        "Logistic Regression"
    };

    public static void main(String[] args) {
        String datasetPath = args.length > 0 ? args[0] : "breast-cancer.arff";

        try {
            // Section 1: dataset loading
            Instances data = loadDataset(datasetPath);

            // Section 2: class assignment
            data.setClassIndex(data.numAttributes() - 1);

            // Section 3: model creation
            Classifier[] classifiers = new Classifier[] {
                new ZeroR(),
                new OneR(),
                new IBk(),
                new NaiveBayes(),
                new J48(),
                new Logistic()
            };

            String[] classifierNames = new String[] {
                "ZeroR",
                "OneR",
                "IBk",
                "Naive Bayes",
                "J48",
                "Logistic Regression"
            };
            System.out.println("==============================================");
            System.out.println("Breast Cancer Classification - 10-Fold CV");
            System.out.println("Dataset: " + datasetPath);
            System.out.println("Class attribute: " + data.classAttribute().name());
            System.out.println("==============================================\n");

            

            for (int i = 0; i < classifiers.length; i++) {
                evaluateAndPrint(classifierNames[i], classifiers[i], data);
            }

            // Section 4 : interactive prediction loop
            Scanner scanner = new Scanner(System.in);
            String selectedModelName = choosePredictionModel(scanner);
            Classifier predictionModel = buildPredictionModel(data, selectedModelName);
            runInteractivePredictionLoop(data, predictionModel, selectedModelName, scanner);

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static Instances loadDataset(String datasetPath) throws Exception {
        try (BufferedReader reader = new BufferedReader(new FileReader(datasetPath))) {
            return new Instances(reader);
        }
    }
    // Section 5: Printing of Model data and evaluation results.
    private static void evaluateAndPrint(String modelName, Classifier classifier, Instances data) throws Exception {
        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(classifier, data, 10, new Random(1));

        double correctlyClassified = evaluation.correct();
        int totalInstances = data.numInstances();

        System.out.println("====================================");
        System.out.println("Model: " + modelName);
        System.out.println("====================================");
        System.out.printf("Correctly Classified Instances: %.0f / %d%n", correctlyClassified, totalInstances);
        System.out.printf("Accuracy: %.2f%%%n", evaluation.pctCorrect());
        System.out.println("Confusion Matrix:");
        System.out.println(evaluation.toMatrixString());
        System.out.println();
    }
    // Selection of model for interactive prediction, with default and input validation.
    private static String choosePredictionModel(Scanner scanner) {
        System.out.println();
        System.out.println("Select a model for interactive predictions:");
        for (int i = 0; i < PREDICTION_MODEL_NAMES.length; i++) {
            System.out.println((i + 1) + ". " + PREDICTION_MODEL_NAMES[i]);
        }

        printPrompt("Enter model number (1-6) or press Enter for default [J48]: ");
        if (!scanner.hasNextLine()) {
            printInfo("No input detected. Using default model: " + DEFAULT_PREDICTION_MODEL_NAME);
            return DEFAULT_PREDICTION_MODEL_NAME;
        }

        String input = scanner.nextLine().trim();
        if (input.isEmpty()) {
            return DEFAULT_PREDICTION_MODEL_NAME;
        }

        try {
            int index = Integer.parseInt(input);
            if (index >= 1 && index <= PREDICTION_MODEL_NAMES.length) {
                return PREDICTION_MODEL_NAMES[index - 1];
            }
        } catch (NumberFormatException e) {
            // Back to default if wala selected model or invalid input.
        }

        printInfo("Invalid choice. Using default model: " + DEFAULT_PREDICTION_MODEL_NAME);
        return DEFAULT_PREDICTION_MODEL_NAME;
    }
    //Buidlding the list of models that the user can choose from for interactive prediction, with error handling for unsupported models.
    private static Classifier buildPredictionModel(Instances data, String predictionModelName) throws Exception {
        Classifier model;
        switch (predictionModelName) {
            case "ZeroR":
                model = new ZeroR();
                break;
            case "OneR":
                model = new OneR();
                break;
            case "IBk":
                model = new IBk();
                break;
            case "Naive Bayes":
                model = new NaiveBayes();
                break;
            case "J48":
                model = new J48();
                break;
            case "Logistic Regression":
                model = new Logistic();
                break;
            default:
                throw new IllegalArgumentException("Unsupported prediction model: " + predictionModelName);
        }

        model.buildClassifier(data);
        return model;
    }
    // Display na ang results of the interactive prediction, Start of displays.
    private static void runInteractivePredictionLoop(Instances data, Classifier predictionModel, String modelName, Scanner scanner) {

        Classifier currentModel = predictionModel;
        String currentModelName = modelName;

        System.out.println("==============================================");
        System.out.println("Interactive Prediction Phase");
        System.out.println("Prediction model: " + currentModelName);
        System.out.println("==============================================");

        // Section 6: interactive prediction loop (loop control)
        boolean keepPredicting = true;

        while (keepPredicting) {
            try {
            // Section 6-a: interactive prediction loop (input collection)
            // Section 6-b: interactive prediction loop (prediction)
                Instance userInstance = collectUserInstance(data, scanner);
                double predictedIndex = currentModel.classifyInstance(userInstance);
                String predictedClass = data.classAttribute().value((int) predictedIndex);

                printInfo("Predicted class: " + predictedClass);
            } catch (Exception e) {
                printInfo("Could not make prediction: " + e.getMessage());
            }

            // Section 6-c: interactive prediction loop (loop control)
            printPrompt("Next action? (yes/y = predict again, change/c = switch model, no/n = exit): ");
            if (!scanner.hasNextLine()) {
                printInfo("No more input detected. Ending program gracefully.");
                break;
            }
            String answer = scanner.nextLine().trim().toLowerCase();

            if (answer.equals("change") || answer.equals("c")) {
                String selectedModelName = choosePredictionModel(scanner);
                try {
                    currentModel = buildPredictionModel(data, selectedModelName);
                    currentModelName = selectedModelName;
                    printInfo("Switched to model: " + currentModelName);
                } catch (Exception e) {
                    printInfo("Could not switch model: " + e.getMessage());
                    printInfo("Continuing with model: " + currentModelName);
                }
                keepPredicting = true;
            } else {
                keepPredicting = answer.equals("yes") || answer.equals("y");
            }
        }

        System.out.println("Program ended. Thank you.");
    }

    private static Instance collectUserInstance(Instances data, Scanner scanner) throws Exception {
        Instance instance = new DenseInstance(data.numAttributes());
        instance.setDataset(data);

        for (int i = 0; i < data.numAttributes(); i++) {
            if (i == data.classIndex()) {
                continue;
            }

            Attribute attr = data.attribute(i);

            if (attr.isNominal()) {
                setNominalValue(instance, attr, scanner);
            } else if (attr.isNumeric()) {
                setNumericValue(instance, attr, scanner);
            } else if (attr.isString()) {
                setStringValue(instance, attr, scanner);
            } else if (attr.isDate()) {
                setDateValue(instance, attr, scanner);
            } else {
                instance.setMissing(attr);
            }
        }

        instance.setMissing(data.classIndex());
        return instance;
    }

    private static void setNominalValue(Instance instance, Attribute attr, Scanner scanner) {
        while (true) {
            boolean rangeNominal = isRangeNominalAttribute(attr);
            if (rangeNominal) {
                printPrompt("Enter value for " + attr.name() + " (numeric, e.g., 45) or leave blank for missing: ");
            } else {
                StringBuilder options = new StringBuilder();
                for (int j = 0; j < attr.numValues(); j++) {
                    options.append(attr.value(j));
                    if (j < attr.numValues() - 1) {
                        options.append(", ");
                    }
                }
                printPrompt("Enter value for " + attr.name() + " [" + options + "] (or leave blank for missing): ");
            }

            if (!scanner.hasNextLine()) {
                instance.setMissing(attr);
                return;
            }
            String input = scanner.nextLine().trim();

            if (input.isEmpty()) {
                instance.setMissing(attr);
                return;
            }

            String normalizedInput = normalizeNominalInput(attr, input);
            int valueIndex = findNominalValueIndexIgnoreCase(attr, normalizedInput);
            if (valueIndex >= 0) {
                instance.setValue(attr, valueIndex);
                return;
            }

            if (rangeNominal) {
                try {
                    double numericInput = Double.parseDouble(normalizedInput);
                    int rangeIndex = findRangeValueIndex(attr, numericInput);
                    if (rangeIndex >= 0) {
                        instance.setValue(attr, rangeIndex);
                        return;
                    }
                    printInfo("Value is outside valid range for " + attr.name() + ". Please try again.");
                    continue;
                } catch (NumberFormatException e) {
                    printInfo("Invalid number. Enter a numeric value like 45.");
                    continue;
                }
            }

            printInfo("Invalid value. Please choose one of the listed options.");
        }
    }

    private static boolean isRangeNominalAttribute(Attribute attr) {
        if (!attr.isNominal() || attr.numValues() == 0) {
            return false;
        }

        for (int i = 0; i < attr.numValues(); i++) {
            String value = attr.value(i);
            if (!value.matches("\\d+-\\d+")) {
                return false;
            }
        }

        return true;
    }

    private static int findRangeValueIndex(Attribute attr, double input) {
        for (int i = 0; i < attr.numValues(); i++) {
            String value = attr.value(i);
            String[] parts = value.split("-");

            if (parts.length != 2) {
                continue;
            }

            try {
                double lower = Double.parseDouble(parts[0]);
                double upper = Double.parseDouble(parts[1]);

                if (input >= lower && input <= upper) {
                    return i;
                }
            } catch (NumberFormatException e) {
                // Ignore malformed range labels.
            }
        }

        return -1;
    }

    private static int findNominalValueIndexIgnoreCase(Attribute attr, String input) {
        for (int i = 0; i < attr.numValues(); i++) {
            if (attr.value(i).equalsIgnoreCase(input)) {
                return i;
            }
        }

        return -1;
    }

    private static String normalizeNominalInput(Attribute attr, String input) {
        String lower = input.toLowerCase();
        int yesIndex = findNominalValueIndexIgnoreCase(attr, "yes");
        int noIndex = findNominalValueIndexIgnoreCase(attr, "no");

        if (yesIndex >= 0 && noIndex >= 0) {
            if (lower.equals("y")) {
                return attr.value(yesIndex);
            }
            if (lower.equals("n")) {
                return attr.value(noIndex);
            }
        }

        // Note: Accepting y/n plus case-insensitive input reduced
        // invalid-entry friction when testing the interactive loop.
        return input;
    }

    private static void setNumericValue(Instance instance, Attribute attr, Scanner scanner) {
        while (true) {
            printPrompt("Enter numeric value for " + attr.name() + " (or leave blank for missing): ");
            if (!scanner.hasNextLine()) {
                instance.setMissing(attr);
                return;
            }
            String input = scanner.nextLine().trim();

            if (input.isEmpty()) {
                instance.setMissing(attr);
                return;
            }

            try {
                double value = Double.parseDouble(input);
                instance.setValue(attr, value);
                return;
            } catch (NumberFormatException e) {
                printInfo("Invalid number. Please enter a valid numeric value.");
            }
        }
    }

    private static void setStringValue(Instance instance, Attribute attr, Scanner scanner) {
        printPrompt("Enter text value for " + attr.name() + " (or leave blank for missing): ");
        if (!scanner.hasNextLine()) {
            instance.setMissing(attr);
            return;
        }
        String input = scanner.nextLine();

        if (input.trim().isEmpty()) {
            instance.setMissing(attr);
            return;
        }

        instance.setValue(attr, input);
    }

    private static void setDateValue(Instance instance, Attribute attr, Scanner scanner) {
        while (true) {
            printPrompt("Enter date for " + attr.name() + " using format " + attr.getDateFormat() + " (or leave blank for missing): ");
            if (!scanner.hasNextLine()) {
                instance.setMissing(attr);
                return;
            }
            String input = scanner.nextLine().trim();

            if (input.isEmpty()) {
                instance.setMissing(attr);
                return;
            }

            try {
                double dateValue = attr.parseDate(input);
                instance.setValue(attr, dateValue);
                return;
            } catch (Exception e) {
                printInfo("Invalid date format. Please try again.");
            }
        }
    }

    private static void printPrompt(String message) {
        System.out.println();
        System.out.print(message);
    }

    private static void printInfo(String message) {
        System.out.println();
        System.out.println(message);
    }


    // Reflective Notes: In our model selection process, we chose J48 as the default 
    // for interactive predictions due to its balance of accuracy and interpretability. 
    // However, we also implemented a flexible system that allows users to select from 
    // multiple models at runtime, including ZeroR, OneR, IBk, Naive Bayes, and Logistic 
    // Regression. This design choice was motivated by the idea of providing users
    // with options while ensuring that the default model is robust enough for general use.
}



def get_example(example_idx: int, max_depth: int | None):

    if max_depth == 1:
        if example_idx == 0:
            example_text = f"\nExample {example_idx + 1} of depth 1:\n"
            example_text += example_features()
            example_text += example_0_depth_1()

        elif example_idx == 1:
            example_text = f"\nExample {example_idx + 1} of depth 1:\n"
            example_text += example_features()
            example_text += example_1_depth_1()

        elif example_idx == 2:
            example_text = f"\nExample {example_idx + 1} of depth 1:\n"
            example_text += example_features()
            example_text += example_2_depth_1()

        elif example_idx == 3:
            example_text = f"\nExample {example_idx + 1} of depth 1:\n"
            example_text += example_features()
            example_text += example_3_depth_1()

        elif example_idx == 4:
            example_text = f"\nExample {example_idx + 1} of depth 1:\n"
            example_text += example_features()
            example_text += example_4_depth_1()

        else:
            raise ValueError(f"Invalid tree index: {example_idx}")
        
    elif max_depth == 2:
        if example_idx == 0:
            example_text = f"\nExample {example_idx + 1} of depth 2:\n"
            example_text += example_features()
            example_text += example_0_depth_2()

        elif example_idx == 1:
            example_text = f"\nExample {example_idx + 1} of depth 1:\n"
            example_text += example_features()
            example_text += example_0_depth_1()

        elif example_idx == 2:
            example_text = f"\nExample {example_idx + 1} of depth 2:\n"
            example_text += example_features()
            example_text += example_1_depth_2()

        elif example_idx == 3:
            example_text = f"\nExample {example_idx + 1} of depth 1:\n"
            example_text += example_features()
            example_text += example_1_depth_1()

        elif example_idx == 4:
            example_text = f"\nExample {example_idx + 1} of depth 2:\n"
            example_text += example_features()
            example_text += example_2_depth_2()

        else:
            raise ValueError(f"Invalid tree index: {example_idx}")

    elif max_depth == 3:
        if example_idx == 0:
            example_text = f"\nExample {example_idx + 1} of depth 3:\n"
            example_text += example_features()
            example_text += example_0_depth_3()

        elif example_idx == 1:
            example_text = f"\nExample {example_idx + 1} of depth 2:\n"
            example_text += example_features()
            example_text += example_0_depth_2()

        elif example_idx == 2:
            example_text = f"\nExample {example_idx + 1} of depth 1:\n"
            example_text += example_features()
            example_text += example_0_depth_1()

        elif example_idx == 3:
            example_text = f"\nExample {example_idx + 1} of depth 3:\n"
            example_text += example_features()
            example_text += example_1_depth_3()

        elif example_idx == 4:
            example_text = f"\nExample {example_idx + 1} of depth 2:\n"
            example_text += example_features()
            example_text += example_1_depth_2()

        else:
            raise ValueError(f"Invalid tree index: {example_idx}")

    elif max_depth == 4:
        if example_idx == 0:
            example_text = f"\nExample {example_idx + 1} of depth 4:\n"
            example_text += example_features()
            example_text += example_0_depth_4()

        elif example_idx == 1:
            example_text = f"\nExample {example_idx + 1} of depth 3:\n"
            example_text += example_features()
            example_text += example_0_depth_3()

        elif example_idx == 2:
            example_text = f"\nExample {example_idx + 1} of depth 2:\n"
            example_text += example_features()
            example_text += example_0_depth_2()

        elif example_idx == 3:
            example_text = f"\nExample {example_idx + 1} of depth 1:\n"
            example_text += example_features()
            example_text += example_0_depth_1()

        elif example_idx == 4:
            example_text = f"\nExample {example_idx + 1} of depth 4:\n"
            example_text += example_features()
            example_text += example_1_depth_4()

        else:
            raise ValueError(f"Invalid tree index: {example_idx}")

    else:

        example_text = f"\nExample {example_idx + 1}"
        if example_idx == 0:
            if max_depth:
                example_text += " of depth 5:\n"
            else:
                example_text += "\n"
            example_text += example_features()
            example_text += example_0_depth_5()

        elif example_idx == 1:
            if max_depth:
                example_text += " of depth 4:\n"
            else:
                example_text += "\n"
            example_text += example_features()
            example_text += example_0_depth_4()

        elif example_idx == 2:
            if max_depth:
                example_text += " of depth 3:\n"
            else:
                example_text += "\n"
            example_text += example_features()
            example_text += example_0_depth_3()

        elif example_idx == 3:
            if max_depth:
                example_text += " of depth 2:\n"
            else:
                example_text += "\n"
            example_text += example_features()
            example_text += example_0_depth_2()

        elif example_idx == 4:
            if max_depth:
                example_text += " of depth 1:\n"
            else:
                example_text += "\n"
            example_text += example_features()
            example_text += example_0_depth_1()

        else:
            raise ValueError(f"Invalid tree index: {example_idx}")


    example_text += "\n\n"
    return example_text


def example_features():
    return """
Features:
gender: Gender refers to the biological sex of the individual, which can have an impact on their susceptibility to diabetes. (0 = Female, 1 = Male, 2 = Other)
age: Age is an important factor as diabetes is more commonly diagnosed in older adults.Age ranges from 0-80 in our dataset. (Years)
hypertension: Hypertension is a medical condition in which the blood pressure in the arteries is persistently elevated. (0 = don’t have hypertension, 1 = have hypertension)
heart_disease: Heart disease is another medical condition that is associated with an increased risk of developing diabetes. (0 = don’t have heart disease, 1 = have heart disease)
smoking_history: Smoking history is also considered a risk factor for diabetes and can exacerbate the complications associated with diabetes. (0 = No Info, 1 = never, 2 = former, 3 = not current, 4 = current, 5 = ever)
bmi: BMI (Body Mass Index) is a measure of body fat based on weight and height. Higher BMI values are linked to a higher risk of diabetes. The range of BMI in the dataset is from 10.16 to 71.55. BMI less than 18.5 is underweight, 18.5-24.9 is normal, 25-29.9 is overweight, and 30 or more is obese.
HbA1c_level: HbA1c (Hemoglobin A1c) level is a measure of a person's average blood sugar level over the past 2-3 months. Higher levels indicate a greater risk of developing diabetes. Mostly more than 6.5% of HbA1c Level indicates diabetes. (Percentage)
blood_glucose_level: Blood glucose level refers to the amount of glucose in the bloodstream at a given time. High blood glucose levels are a key indicator of diabetes.

Target:
diabetes: Presence of diabetes. (0 = No, 1 = Yes)
"""


def example_0_depth_1():
    return """
def predict(X: dict):
    nodes = 1 * [None]
    nodes[0] = X["HbA1c_level"] <= 6.7
    
    if nodes[0]:
      prediction = 0
    else:
      prediction = 1
    return prediction, nodes
"""

def example_1_depth_1():
    return """
def predict(X: dict):
    nodes = 1 * [None]
    nodes[0] = X["blood_glucose_level"] <= 210
    
    if nodes[0]:
      prediction = 0
    else:
      prediction = 1
    return prediction, nodes
"""

def example_2_depth_1():
    return """
def predict(X: dict):
    nodes = 1 * [None]
    nodes[0] = X["hypertension"] <= 0.5
    
    if nodes[0]:
      prediction = 0
    else:
      prediction = 1
    return prediction, nodes
"""

def example_3_depth_1():
    return """
def predict(X: dict):
    nodes = 1 * [None]
    nodes[0] = X["blood_glucose_level"] <= 190
    
    if nodes[0]:
      prediction = 0
    else:
      prediction = 1
    return prediction, nodes
"""

def example_4_depth_1():
    return """
def predict(X: dict):
    nodes = 1 * [None]
    nodes[0] = X["bmi"] > 25
    
    if nodes[0]:
      prediction = 1
    else:
      prediction = 0
    return prediction, nodes
"""


def example_0_depth_2():
    return """
def predict(X: dict):
    nodes = 2 * [None]
    nodes[0] = X["HbA1c_level"] <= 6.7
    nodes[1] = X["blood_glucose_level"] <= 210
    
    if nodes[0]:
        if nodes[1]:
            prediction = 0
        else:
            prediction = 1
    else:
        prediction = 1

    return prediction, nodes
"""


def example_1_depth_2():
    return """
def predict(X: dict):
    nodes = 3 * [None]
    nodes[0] = X["HbA1c_level"] <= 6.7
    nodes[1] = X["blood_glucose_level"] <= 210
    nodes[2] = X["age"] <= 54.5

    if nodes[0]:
        if nodes[1]:
            prediction = 0
        else:
            prediction = 1
    else:
        if nodes[2]:
            prediction = 0
        else:
            prediction = 1
    return prediction, nodes
"""


def example_2_depth_2():
    return """
def predict(X: dict):
    nodes = 3 * [None]
    nodes[0] = X["petal_length"] <= 3.0
    nodes[1] = X["sepal_width"] <= 2.5
    nodes[2] = X["petal_width"] <= 0.8
    
    if nodes[0]:
      if nodes[1]:
          prediction = 1
      else:
          prediction = 0
    else:
      if nodes[2]:
          prediction = 0
      else:
          prediction = 1
    return prediction, nodes
"""

def example_0_depth_3():
    return """
def predict(X: dict):
    nodes = 3 * [None]
    nodes[0] = X["petal_length"] <= 2.5
    nodes[1] = X["sepal_width"] <= 3.0
    nodes[2] = X["petal_width"] <= 1.5
    
    if nodes[0]:
      prediction = 2
    else:
      if nodes[1]:
          if nodes[2]:
              prediction = 0
          else:
              prediction = 1
      else:
          prediction = 2
    return prediction, nodes
"""

def example_1_depth_3():
    return """
def predict(X: dict):
    nodes = 5 * [None]
    nodes[0] = X["sepal_length"] <= 5.5
    nodes[1] = X["petal_width"] <= 1.7
    nodes[2] = X["petal_length"] <= 4.8
    nodes[3] = X["sepal_width"] <= 3.0
    nodes[4] = X["sepal_width"] > 2.0
    
    if nodes[0]:
      if nodes[1]:
        if nodes[2]:
            prediction = 2
        else:
            prediction = 0
      else:
        if nodes[3]:
            prediction = 2
        else:
            prediction = 0
    else:
      if nodes[4]:
          prediction = 1
      else:
          prediction = 0
    return prediction, nodes
"""

def example_0_depth_4():
    return """
def predict(X: dict):
    nodes = 4 * [None]
    nodes[0] = X["petal_length"] <= 2.5
    nodes[1] = X["sepal_width"] > 3.1
    nodes[2] = X["petal_width"] <= 1.3
    nodes[3] = X["sepal_length"] > 4.8

    if nodes[0]:
        if nodes[1]:
            if nodes[2]:
                if nodes[3]:
                    prediction = 1
                else:
                    prediction = 2
            else:
                if nodes[3]:
                    prediction = 0
                else:
                    prediction = 1
        else:
            prediction = 2
    else:
        if nodes[2]:
            if nodes[3]:
                prediction = 0
            else:
                prediction = 1
        else:
            prediction = 2
    return prediction, nodes
"""

def example_1_depth_4():
    return """
def predict(X: dict):
    nodes = 4 * [None]
    nodes[0] = X["sepal_length"] <= 5.5
    nodes[1] = X["petal_width"] > 1.4
    nodes[2] = X["petal_length"] <= 3.7
    nodes[3] = X["sepal_width"] > 2.9

    if nodes[0]:
        if nodes[1]:
            if nodes[2]:
                if nodes[3]:
                    prediction = 0
                else:
                    prediction = 2
            else:
                if nodes[3]:
                    prediction = 1
                else:
                    prediction = 0
        else:
            prediction = 1
    else:
        if nodes[2]:
            if nodes[3]:
                prediction = 1
            else:
                prediction = 2
        else:
            prediction = 0
    return prediction, nodes
"""

def example_0_depth_5():
    return """
def predict(X: dict):
    nodes = 5 * [None]
    nodes[0] = X["petal_length"] <= 2.5
    nodes[1] = X["sepal_width"] > 3.1
    nodes[2] = X["petal_width"] <= 1.3
    nodes[3] = X["sepal_length"] > 4.8
    nodes[4] = X["petal_length"] > 1.1

    if nodes[0]:
        if nodes[1]:
            if nodes[2]:
                if nodes[3]:
                    if nodes[4]:
                        prediction = 1
                    else:
                        if nodes[3]:
                            prediction = 0
                        else:
                            prediction = 2
                else:
                    prediction = 0
            else:
                if nodes[4]:
                    prediction = 2
                else:
                    prediction = 1
        else:
            prediction = 2
    else:
        if nodes[2]:
            prediction = 0
        else:
            prediction = 1
    return prediction, nodes
"""


#To be updated


def format_model_inputs(input_dict):
    age = int(input_dict['age'])
    sex = input_dict['sex']
    bmi = float(input_dict['bmi'])
    children = int(input_dict['children'])
    smoker = input_dict['smoker']
    region = input_dict['region']

    return [age, sex, bmi, children, smoker, region]


def validate(input_dict):
    errors = []

    required_fields = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

    for field in required_fields:

        if field not in input_dict:
            errors.append(f'{field} not found in request.')

        if field in ['age', 'children'] and type(input_dict[field]) != int \
                and not input_dict[field].isnumeric():
            errors.append(f'Age and children fields must be int type.')

        elif field == 'bmi' and type(input_dict[field]) != float:
            try:
                float(input_dict['bmi'])
            except ValueError:
                errors.append(f'bmi field must be numeric.')

    return errors

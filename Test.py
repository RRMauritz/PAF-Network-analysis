import numpy as np

exclude = {'Giorgio': [3, 12, 4], 'Janet': [14, 15, 16], 'Huan': [2, 13, 17], 'Jop': [8, 1, 18], 'Rutger': [6, 9, 10],
           'Thomas': [11, 5, 7]}

available_problems = [i for i in range(1, 19)]

for student in exclude:
    sample_set = list(set(available_problems) - set(exclude[student]))
    print(sample_set)
    review = np.random.choice(sample_set,3,False)
    print(student, ' reviews ', review)
    available_problems = list(set(available_problems)-set(review))
    print(available_problems)



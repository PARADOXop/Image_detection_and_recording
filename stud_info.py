
try:
    n = int(input('enter the number of students in school:-'))
    if n == 0:
        print('stduent in the school can\'t be zero')
        raise ValueError()
except ValueError :
    if n == 0:
        print('how the hell are you')
except:
    print('something went wrong')
else:
    A = list(range(1, n + 1))


    class school:
        name = 'MODEL ENGLISH SCHOOL'

        def get_info(self):
            self.name = input('enter name of the student:')
            self.rollno = int(input('enter rollno of the student'))
            self.age = int(input('enter the age'))

        def show_info(self):
            print('***************************************************************')
            print('name of the student :', self.name)
            print('RollNo. of the student is :', self.rollno)
            print('RollNo. of the student is :', self.age)
            print('***************************************************************')

        @classmethod
        def school(cls):
            print(school.name)


    print('***************************', school.name, '*******************************')

    for i in range(1, n + 1):
        S = 'S' + str(i)
        print('for student', S)
        S = school()
        S.get_info()
        A.insert(i, S)
    print('do you want to print info about student:')
    print('1 for yess and 2 for no')
    x =int(input('enter the value'))
    if x == 1:
        for j in range(1, n + 1):
            A[j].show_info()
    else:
        print('go back home kid what are you doi\'n here')
    print('*******************************', school.name, '******************************')

finally :
    print('we are equally fucked up')
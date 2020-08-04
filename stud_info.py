import array
import numpy
n = int(input('enter the number of students in school:-'))
A =numpy.zeros(n)
for e in A:
    print(e)

for i in range(1, n+1):
    print(i)
class school :
    name = 'MODEL ENGLISH SCHOOL'

    def get_info(self):
        self.name = input('enter name of the student:')
        self.rollno = int(input('enter rollno of the student'))
        self.age = int(input('enter the age'))
    def show_info(self):
        print('***************************************************************')
        print(self.name)
        print(self.rollno)
        print(self.age)
        print('***************************************************************')

    @classmethod
    def school(cls):
        print(school.name)

print(school.name)

for i in range(1, n+1):
  S = 'S' + str(i)
  print('for student',S)
  S = school()
  S.get_info()
  for j in range(1, n+1):
        E = 'E' + str(j)
        E = S
        E.show_info()
print(school.name)
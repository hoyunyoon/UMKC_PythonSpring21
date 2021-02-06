class Employee:
    count = 0
    Tot_Salary = 0

    def __init__(self, name, family, salary, dept):
        self.name = name
        self.family = family
        self.salary = salary
        self.dept = dept
        Employee.count += 1
        Employee.Tot_Salary += salary

    def employee_count(self):

        print("total number of employees", Employee.count)

    def average_salary(self):

        avg_sal = Employee.Tot_Salary / Employee.count
        print("average salary:", avg_sal)

    def samp_function(self):
        print('calling base class member function')


class Full_time_employee(Employee):
    def __init__(self):
        print('Full time employee(sub class)')

if __name__ == '__main__':
    num_of_emp = int(input("No of employees:"))
    for i in range(num_of_emp):
        nam = input("name:")
        fam = input("family:")
        sal = float(input("salary:"))
        dep = input("dept:")
        BaseClass = Employee(nam,fam,sal,dep)
    Employee_Details = Full_time_employee()
    Employee_Details.employee_count()
    Employee_Details.average_salary()
    BaseClass.samp_function()

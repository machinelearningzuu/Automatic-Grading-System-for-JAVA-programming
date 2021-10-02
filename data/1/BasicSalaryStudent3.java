package question.one;

import java.util.Scanner;

public class BasicSalary {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		String EmployeeNumber;
		double basicSalary;
		double netSalary;
		
		Scanner s = new Scanner(System.in);
		
		System.out.println("Enter Employee Number : ");
		EmployeeNumber = s.nextLine();
		
		System.out.println("Enter Basic Salary : ");
		basicSalary = s.nextDouble();
		
		double salAdd = 110.00/100.00;
		
		netSalary = basicSalary*salAdd;
		
		System.out.println(EmployeeNumber + " Your Net Salary is Rs : "+netSalary);

	}

}

package question.one;

import java.util.Scanner;

public class BasicSalary {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		String empno;
		double basicSal;
		double netSal;
		
		Scanner s = new Scanner(System.in);
		
		System.out.println("Enter Employee Number : ");
		empno = s.nextLine();
		
		System.out.println("Enter Basic Salary : ");
		basicSal = s.nextDouble();
		
		double salAdd = 110.00/100.00;
		
		netSal = basicSal*salAdd;
		
		System.out.println(empno + " Your Net Salary is Rs : "+netSal);

	

}

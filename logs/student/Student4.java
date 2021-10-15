package question.one;

import java.util.Scanner;

public class BasicSalary {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		String empno;
		int basicSal;
		int netSal;
		
		Scanner s = new Scanner(System.in);
		
		System.out.println("Enter Employee Number : ");
		empno = s.nextLine();
		
		System.out.println("Enter Basic Salary : ");
		basicSal = s.nextInt();
		
		int salAdd = 110/100;
		
		netSal = basicSal*salAdd;
		
		System.out.println(empno + " Your Net Salary is Rs : "+netSal);

	}

}

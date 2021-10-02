package question.three;

import java.util.Scanner;

public class Student {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		String Studentname;
		int Studentage;
		double fee;
		
		Scanner s = new Scanner(System.in);
		
		System.out.println("Enter Your Name : ");
		Studentname = s.nextLine();
		
		System.out.println("Enter Your Age : ");
		Studentage = s.nextInt();
		
		System.out.println("Enter Your Course Fee : ");
		fee = s.nextDouble();
		
		System.out.println(
				Studentname + 
				Studentage + 
				fee
				);
	}

}

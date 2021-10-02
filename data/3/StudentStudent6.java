package question.three;

import java.util.Scanner;

public class Student {
 {

		String name;
		int age;
		double fee;
		
		Scanner s = new Scanner(System.in);
		
		System.out.println("Enter Your Name : ");
		name = s.nextLine();
		
		System.out.println("Enter Your Age : ");
		age = s.nextInt();
		
		System.out.println("Enter Your Course Fee : ");
		fee = s.nextDouble();
		
		System.out.println(
				name + 
				age + 
				fee
				);
	}

}

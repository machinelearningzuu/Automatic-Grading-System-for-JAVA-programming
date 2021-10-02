package question.four;

import java.util.Scanner;

public class Grades {

 {
	
		
		String name;
		float avg;
		String grade;
		
		Scanner s = new Scanner(System.in);
		
		System.out.println("Enter your name : ");
		name = s.nextLine();
		
		System.out.println("Enter your Average : ");
		avg = s.nextFloat();
		
		if (avg > 80) {
			grade = "Distinction";
		} else if (avg > 70){
			grade = "Credit";
		} else if (avg > 60) {
			grade = "Simple Pass";
		} else {
			grade = "Fail";
		}
		
		System.out.println(
				name +" " +grade
				);

	}

}

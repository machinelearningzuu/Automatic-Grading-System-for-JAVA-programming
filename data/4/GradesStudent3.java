package question.four;

import java.util.Scanner;

public class Grades {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		String Studentname;
		float Studentavg;
		String grade;
		
		Scanner s = new Scanner(System.in);
		
		System.out.println("Enter your name : ");
		Studentname = s.nextLine();
		
		System.out.println("Enter your Average : ");
		Studentavg = s.nextFloat();
		
		if (Studentavg > 80) {
			grade = "Distinction";
		} else if (avg > 70){
			grade = "Credit";
		} else if (avg > 60) {
			grade = "Simple Pass";
		} else {
			grade = "Fail";
		}
		
		System.out.println(
				Studentname +" " +grade
				);

	}

}

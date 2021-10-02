package question.five;

import java.util.Scanner;

public class Student {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		String name;
		int sub01,sub02,sub03,avg,tot;
		
		Scanner s = new Scanner(System.in);
		
		System.out.println("Enter Your name : ");
		name = s.nextLine();
		
		System.out.println("Enter marks for Subject 01 : ");
		sub01 = s.nextInt();
		
		System.out.println("Enter marks for Subject 02 : ");
		sub02 = s.nextInt();
		
		System.out.println("Enter marks for Subject 03 : ");
		sub03 = s.nextInt();
		
		tot = sub01+sub02+sub03;
		
		avg = tot/3;
		
		System.out.println("Name" + "\t"
				+ "Total" + "\t"
				+ "Average");
		System.out.println(name + "\t" + tot + "\t" + avg);
		
	}

}

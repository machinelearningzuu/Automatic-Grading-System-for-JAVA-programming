package question.two;

import java.util.Scanner;

public class Name {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		int f_name;
		int l_name;
		
		Scanner s = new Scanner(System.in);
		
		System.out.println("Enter Your First Name : ");
		f_name = s.nextInt();
		
		System.out.println("Enter Your Last Name : ");
		l_name = s.nextInt();
		
		System.out.println(f_name + " " +l_name);

	}

}

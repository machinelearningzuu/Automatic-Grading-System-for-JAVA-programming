package question.two;

import java.util.Scanner;

public class Name {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		String firstname;
		String lastname;
		
		Scanner s = new Scanner(System.in);
		
		System.out.println("Enter Your First Name : ");
		firstname = s.nextLine();
		
		System.out.println("Enter Your Last Name : ");
		lastname = s.nextLine();
		
		System.out.println(firstname + " " +lastname);

	}

}

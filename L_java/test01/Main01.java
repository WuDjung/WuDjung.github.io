import java.util.Scanner;
public class Main01 {
        public static void main(String[] args) {
            Scanner in = new Scanner(System.in);

            for(int i=1;i<=3;i++) {
                System.out.print("what's your name? ");
                String name = in.nextLine();

                System.out.print("and your age is? ");
                int age = in.nextInt();
                in.nextLine();

                System.out.printf("welcome %s , and your age will be %d next year\n", name, age+1);
            }

            in.close();
        }
}
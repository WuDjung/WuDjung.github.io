import java.util.Scanner;

public class Main02 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);

        String line = in.nextLine();          // 整行读取
        String[] parts = line.trim().split("\\s+"); // 按空白切分
        int[] arr = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            arr[i] = Integer.parseInt(parts[i]);
        }

        double ave = average(arr);
        int max = max(arr);

        System.out.printf("Average is %.2f and Max is %d%n", ave, max);

        in.close();
    }

    public static double average(int[] arr) {
        double s = 0;
        for (int value : arr) s += value;   // 增强 for 循环
        return s / arr.length;              // arr.length 是属性
    }

    public static int max(int[] arr) {
        int m = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > m) m = arr[i];
        }
        return m;
    }
}

import java.util.Scanner;

public class Main04 {
    public static void main(String[] args) {
        WordStats stats = new WordStats();
        // 用while循环读取每行（Java 1.5+ 都支持）
        Scanner scanner = new Scanner(System.in);
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            stats.addLine(line);
        }
        scanner.close(); // 手动关闭资源
        
        stats.printCount();
        System.out.println("unique words: " + stats.uniqueCount());
    }
}

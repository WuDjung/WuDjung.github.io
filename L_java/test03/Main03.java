import java.util.*;
import java.util.stream.Collectors;

public class Main03 {
    public static void main(String[] args) {
        // 使用try-with-resources自动关闭Scanner，无需手动调用close()
        try (Scanner scanner = new Scanner(System.in)) {
            // 读取输入并处理空值/全空格情况
            String input = scanner.nextLine().trim();
            if (input.isEmpty()) {
                System.out.println("输入为空，请重新输入单词");
                return;
            }

            // 分割为单词数组并转换为List
            String[] inputWords = input.split("\\s+");
            List<String> words = new ArrayList<>(Arrays.asList(inputWords));

            // .stream()为输入流，.collect()为流的 “终止操作”，Collectors.groupingBy为收集器
            Map<String, Long> wordFrequency = words.stream()
                    .collect(Collectors.groupingBy(
                            word -> word,  // 按单词本身分组,  分类函数:按什么标准分组
                            Collectors.counting()  // 统计每组数量， 下游收集器:对每个组内的元素做什么操作
                    ));

            // 按字母顺序输出统计结果（保持TreeSet去重排序特性）
            new TreeSet<>(words).forEach(          //匿名对象直接调用方法,treeSet只用一次
                word -> System.out.println(word + ": " + wordFrequency.get(word))        //.grt()方法获取对应键的值
            );

            // 优化：直接对List排序（逻辑更直观）
            Collections.sort(words);
            System.out.println("after sorted: " + words);
        } // Scanner在这里会自动关闭，无需手动处理
    }
}


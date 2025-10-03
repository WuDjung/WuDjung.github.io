import java.util.*;

public class WordStats {
    private final Map<String, Integer> count = new HashMap<>();

    public void addLine(String line) {
        if (line == null || line.trim().isEmpty()) {
            return; // 跳过空行
        }
        // 分割单词并处理标点符号和空字符串
        String[] words = line.trim().split("\\s+");
        for (String word : words) {
            // 去除单词前后的标点符号，只保留字母和数字
            String cleanedWord = word.replaceAll("^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "").trim();
            if (!cleanedWord.isEmpty()) { // 过滤处理后为空的字符串
                // 转为小写，使统计不区分大小写（如Hello和hello视为同一个单词）
                cleanedWord = cleanedWord.toLowerCase();
                count.put(cleanedWord, count.getOrDefault(cleanedWord, 0) + 1);
            }
        }
    }

    public void printCount() {
        // 使用TreeMap按字母顺序排序输出
        new TreeMap<>(count).forEach((word, count) -> 
            System.out.printf("%s: %d%n", word, count)
        );
    }

    public int uniqueCount() {
        return count.size();
    }
}

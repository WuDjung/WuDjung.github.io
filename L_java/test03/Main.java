import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        
        String[] tmp = in.nextLine().trim().split("\\s+");
        List<String> words = new ArrayList<>(Arrays.asList(tmp));

        Map<String, Integer> count = new HashMap<>();
        for (String w : words) {
            count.put(w, count.getOrDefault(w,0) + 1);
        }

        Set<String> uniq = new TreeSet<>(words);

        for (String w : uniq) {
            System.out.println(w + ": " + count.get(w));
        }

        in.close();
    }
}
public interface ShowMessage {
    void 显示商标(String s)；
}

public class 01 {
    public static void main(String[] args) {
       ShowMessage sm;
       sm(s)-> {
        System.out.println("sss");
        System.out.println(s);
        System.out.println("tvt");
       }
       sm.显示商标(s:"长城牌电器");
       sm=(s)->{
        System.out.println("aaa");
        System.out.println(s);
        System.out.println("QAQ");
       }
       sm.显示商标("化为");
    }
}
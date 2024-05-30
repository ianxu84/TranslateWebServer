$(document).ready(function(){
    $("#submit").click(function(event){
        event.preventDefault(); // 停用預設事件

        text = $("#text1").val(); // 檢查是否有輸入翻譯字
        if (text.length > 0){
            $("#submit").text("翻譯中..."); // 替換按鈕文字增強使用者體驗
            $.ajax({
                type: "POST",
                url: "/translation",
                data: {'text': text }, // 傳給Flask API
                success: function(res){
                    if ( $("#text2").length > 0 ){
                        $("#text2").text(res)
                        console.log(res);
                    }
                    else{
                        $(".translation").append("<textarea class='text' id='text2' name='text' readonly>" + res + "</textarea>");
                    }
                    $("#submit").text("翻譯");
                }
            })
        }
    })
})
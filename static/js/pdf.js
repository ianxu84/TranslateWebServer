$(document).ready(function(){
    /* 顯示列表 */
    update_list();
    update_glossaries();

    /* 上傳按鈕 */
    $("#btn_upload_deepl").on("click", function(){
        var fileInput = $("#file_input_deepl")
        fileInput.click()
    })

    $("#btn_upload_deepl_image").on("click", function(){
        var fileInput = $("#file_input_deepl_image")
        fileInput.click()
    })

    /* 上傳檔案觸發 */
    $("#file_input_deepl").on("change", function(){
        var file = this.files[0]
        if (!file){
            console.log("未選擇檔案")
        }
        else if (file.type === "application/pdf"){
            var glossary = $("#glossaries").val();
            /* 執行時顯示動畫 */
            $(".loading_container")[0].style.display = "block";
            upload_pdf('deepl', '');
            translate_pdf(file.name, 'deepl', 'none', glossary);
            
        }
        else {
            $("#file_input_deepl").text("")
            alert("檔案類型錯誤!")
        }
    })

    $("#file_input_deepl_image").on("change", function(){
        var file = this.files[0]
        if (!file){
            console.log("未選擇檔案")
        }
        else if (file.type === "application/pdf"){
            $("#page_select")[0].style.display = "block";
            $("#btn_upload_image_check")[0].style.display = "block";
            $("#select_message")[0].style.display = "block";
        }
        else {
            alert("檔案類型錯誤!")
        }
    })

    // 當focus在分頁輸入框時點enter 自動觸發"確認上傳"按鈕
    $("#page_select").on("keypress", function(e){
        if (e.which === 13){
            $("#btn_upload_image_check").click();
        }
    });

    $("#btn_upload_image_check").on("click", function(){
        var file = $("#file_input_deepl_image")
        var ocr = $("#page_select").val();
        var regex = /^[0-9,-]+$/;
        
        if (ocr.length == 0){
            ocr = "all"; // 如果沒有輸入值則預設全部翻譯
        }
        else if (!regex.test(ocr)){  // 如果輸入包含非0到9、逗號、減號
            file.val('');
            reset_page();  // 重置頁面元素狀態
            alert("輸入不符合規範!");
        }
        if (file.length > 0 && (ocr === 'all' || regex.test(ocr))){
            /* 執行時顯示動畫與 */
            var glossary = $("#glossaries").val();
            var fileName = file[0].files[0].name
            
            $(".loading_container")[0].style.display = "block";
            $.ajax({
                type: "DELETE",
                url: '/delete/pdf/' + fileName,
                success: function(res){
                    console.log('Delete PDF Success')
                    upload_pdf('deepl','_image');
                    translate_pdf(fileName, 'deepl', ocr, glossary);
                },
                error: function(xhr, status, error){
                    console.error('Delete PDF Error')
                }
            })
        }
    })
})

function update_list(){
    /* 更新列表 */
    $.ajax({
        type: "GET",
        url: "/get/pdf/translated",
        success: function(res){
            /* 清空原有列表 */
            var tableBody = $('#list_table_translated tbody');
            tableBody.empty();
            if (res.length > 0) {
                /* 寫入新列表 */
                var newTableHeader = $('<tr>').addClass("pdf_list_translated list_th");
                newTableHeader.append($('<td>').addClass("pdf_list_translated list_td").text("已翻譯檔案"));
                tableBody.append(newTableHeader);
                $.each(res, function(index, item){
                    var newRow = $('<tr>').addClass("pdf_list_translated list_tr");
                    var newTd = $('<td>').addClass("pdf_list_translated list_td");
                    var newLink = $('<a>').addClass("pdf_list_translated list_a").attr("href", '/download/pdf/'+item).text(item);
                    var item2 = item.replace("_zh.pdf", "_zh_en.pdf");
                    var newLink = $('<a>').addClass("pdf_list_translated list_a").attr("href", '/download/pdf/'+item).text(item);
                    var newLink2 = $('<a>').addClass("pdf_list_translated list_a").attr("href", '/download/pdf/'+item2).text(item2);
                    newTd.append(newLink, newLink2);
                    newRow.append(newTd);
                    tableBody.append(newRow);
                });
            }
            else {
                tableBody.append('<p class="message"><h2>無翻譯資料</h2></p>')
            }
        }
    });
}

function reset_page(){
    $("#page_select")[0].value = "";
    $("#page_select")[0].style.display = "none";
    $("#btn_upload_image_check")[0].style.display = "none";
    $("#select_message")[0].style.display = "none";
}

function update_glossaries(){
    /* 更新詞彙表列表 */
    $.ajax({
        type: "GET",
        url: "/get/glossaries",
        success: function(res){
            var glossaries = $("#glossaries")
            // console.log(res)
            if (res.length > 0){
                $.each(res, function(index, item){
                    // console.log(item)
                    glossaries.append('<option value="'+ item[0] +'">'+ item[1] +'</option>')
                });
            }
        }
    })
}

function upload_pdf(engine, image){
    // engine: deepl, chatgpt
    var file = $("#file_input_" + engine + image)[0];
    var formData = new FormData();
    formData.append('file', file.files[0]);
    console.log(file.files[0])
    if(file.files[0].type === "application/pdf"){
        $.ajax({
            type: "POST",
            url: "/upload/pdf",
            data: formData,
            processData: false,
            contentType: false,
            timeout: 120000,
            success: function(res){
                console.log('Upload Success')
            },
            error: function(error){
                console.error(error);
                alert("Upload Failed");
            }
        })
    }
    else {
        alert("檔案類型錯誤!");
    }
}

function translate_pdf(fileName, engine, ocr, glossary){
    console.log('Translate start')
    $.ajax({
        type: "POST",
        url: "/translate/pdf/" + engine + "/" + fileName + "/" + ocr + "/" + glossary,
        timeout: 180000,
        success: function(res){
            if(res==='OK'){
                console.log('Translate Success');
                update_list();
                window.open('download/pdf/' + fileName.replace(/\.pdf$/, "_zh.pdf"));
                window.open('download/pdf/' + fileName.replace(/\.pdf$/, "_zh_en.pdf"));

                // 清除值
                $("#file_input_" + engine).val('');
                $("#file_input_" + engine + "_image").val('');
                // $.ajax({
                //     type: "DELETE",
                //     url: '/delete/pdf/' + fileName,
                //     success: function(res){
                //         console.log('Delete PDF Success')
                //     },
                //     error: function(xhr, status, error){
                //         console.error('Delete PDF Error')
                //     }
                // })
            }
            else {
                // 清除值
                $("#file_input_" + engine).val('');
                $("#file_input_" + engine + "_image").val('');
                alert(res);
            }
        },
        error: function(e){
            console.log("error:",e)
        },
        complete: function(){
            $(".loading_container")[0].style.display = "none";
            $("#page_select")[0].value = "";
            $("#page_select")[0].style.display = "none";
            $("#btn_upload_image_check")[0].style.display = "none";
            $("#select_message")[0].style.display = "none";
        }
    })
}
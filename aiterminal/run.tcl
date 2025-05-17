#!/usr/bin/tclsh

# Verilog 模擬與分析一鍵執行腳本

# 顯示標題
puts "\n========================================="
puts "   Verilog 4位元計數器模擬與分析系統"
puts "=========================================\n"

# 設置工作目錄
set work_dir "verilog_workspace"
file mkdir $work_dir

# 執行 Verilog 開發階段
puts "階段 1: 執行 Verilog 開發者代理..."
puts "----------------------------------------"

if {[catch {exec python verilog_developer.py} result]} {
    puts "\n❌ Verilog 開發階段失敗!"
    puts "錯誤信息:"
    puts $result
    puts "\n請修正錯誤後重試。"
    exit 1
} else {
    puts $result
    puts "\n✅ Verilog 開發階段成功完成!"
}

# 確認模擬結果存在
set output_file [file join $work_dir "simulation_output.txt"]
if {![file exists $output_file]} {
    puts "\n❌ 找不到模擬輸出文件: $output_file"
    puts "請檢查 Verilog 開發者代理是否正確運行。"
    exit 1
}

# 顯示模擬結果
puts "\n模擬結果: (來自 $output_file)"
puts "----------------------------------------"
if {[catch {open $output_file r} fileId]} {
    puts "無法打開模擬輸出文件: $fileId"
} else {
    set sim_content [read $fileId]
    close $fileId
    puts $sim_content
    puts "----------------------------------------"
}

# 等待一會兒
puts "\n準備進行結果分析...(3秒)"
after 3000

# 執行結果分析階段
puts "\n階段 2: 執行結果分析代理..."
puts "----------------------------------------"

if {[catch {exec python results_analyzer.py} result]} {
    puts "\n❌ 結果分析階段失敗!"
    puts "錯誤信息:"
    puts $result
    puts "\n請修正錯誤後重試。"
    exit 1
} else {
    puts $result
    puts "\n✅ 結果分析階段成功完成!"
}

# 確認分析報告存在
set report_file [file join $work_dir "analysis_report.txt"]
if {![file exists $report_file]} {
    puts "\n❌ 找不到分析報告文件: $report_file"
    exit 1
}

# 打開分析報告
puts "\n\n========================================="
puts "   分析報告內容"
puts "=========================================\n"

if {[catch {open $report_file r} fileId]} {
    puts "無法打開報告文件: $fileId"
} else {
    set report_content [read $fileId]
    close $fileId
    puts $report_content
}

puts "\n========================================="
puts "   流程完成!"
puts "   所有文件保存在 $work_dir 目錄中"
puts "=========================================\n"
// =============================================================================
// Package: neurisc_pkg
// Description: UVM Package for NeuroRISC Verification Environment
// 
// Contains:
// - Transaction classes
// - Sequence items
// - Reference models
// - Scoreboards
// - Coverage collectors
// =============================================================================

package neurisc_pkg;
    
    import uvm_pkg::*;
    `include "uvm_macros.svh"
    
    // ==========================================================================
    // Transaction Classes
    // ==========================================================================
    
    // MAC Unit Transaction
    class mac_transaction extends uvm_sequence_item;
        rand bit signed [7:0] weight_in;
        rand bit signed [7:0] input_in;
        rand bit              enable;
        rand bit              clear_acc;
        bit signed [19:0]     accumulator;
        bit signed [7:0]      weight_out;
        bit signed [7:0]      input_out;
        
        `uvm_object_utils_begin(mac_transaction)
            `uvm_field_int(weight_in, UVM_ALL_ON)
            `uvm_field_int(input_in, UVM_ALL_ON)
            `uvm_field_int(enable, UVM_ALL_ON)
            `uvm_field_int(clear_acc, UVM_ALL_ON)
            `uvm_field_int(accumulator, UVM_ALL_ON)
        `uvm_object_utils_end
        
        function new(string name = "mac_transaction");
            super.new(name);
        endfunction
    endclass
    
    // Matrix Transaction
    class matrix_transaction extends uvm_sequence_item;
        rand bit signed [7:0] matrix_a [0:7][0:7];
        rand bit signed [7:0] matrix_b [0:7][0:7];
        bit signed [19:0]     result [0:7][0:7];
        
        `uvm_object_utils(matrix_transaction)
        
        function new(string name = "matrix_transaction");
            super.new(name);
        endfunction
        
        // Constraints for corner cases
        constraint zero_matrix_c {
            foreach(matrix_a[i,j]) matrix_a[i][j] == 0;
        }
        
        constraint identity_matrix_a_c {
            foreach(matrix_a[i,j]) {
                if (i == j) matrix_a[i][j] == 1;
                else matrix_a[i][j] == 0;
            }
        }
        
        constraint max_values_c {
            foreach(matrix_a[i,j]) matrix_a[i][j] == 127;
            foreach(matrix_b[i,j]) matrix_b[i][j] == 127;
        }
        
        constraint min_values_c {
            foreach(matrix_a[i,j]) matrix_a[i][j] == -128;
            foreach(matrix_b[i,j]) matrix_b[i][j] == -128;
        }
    endclass
    
    // Activation Transaction
    class activation_transaction extends uvm_sequence_item;
        rand bit [1:0]          func_select;
        rand bit signed [15:0]  data_in;
        bit signed [15:0]       data_out;
        bit signed [15:0]       expected_out;
        
        `uvm_object_utils_begin(activation_transaction)
            `uvm_field_int(func_select, UVM_ALL_ON)
            `uvm_field_int(data_in, UVM_ALL_ON)
            `uvm_field_int(data_out, UVM_ALL_ON)
        `uvm_object_utils_end
        
        function new(string name = "activation_transaction");
            super.new(name);
        endfunction
    endclass
    
    // DMA Transaction
    class dma_transaction extends uvm_sequence_item;
        rand bit [31:0] src_addr;
        rand bit [31:0] dst_addr;
        rand bit [15:0] transfer_size;
        rand bit        mode_2d;
        rand bit [15:0] row_count;
        rand bit [15:0] col_count;
        rand bit [15:0] src_stride;
        rand bit [15:0] dst_stride;
        bit             done;
        
        `uvm_object_utils_begin(dma_transaction)
            `uvm_field_int(src_addr, UVM_ALL_ON)
            `uvm_field_int(dst_addr, UVM_ALL_ON)
            `uvm_field_int(transfer_size, UVM_ALL_ON)
            `uvm_field_int(mode_2d, UVM_ALL_ON)
        `uvm_object_utils_end
        
        function new(string name = "dma_transaction");
            super.new(name);
        endfunction
    endclass
    
    // ==========================================================================
    // Reference Models
    // ==========================================================================
    
    // MAC Reference Model
    class mac_reference_model;
        static function bit signed [19:0] compute_mac(
            bit signed [7:0] weight,
            bit signed [7:0] input_val,
            bit signed [19:0] prev_acc,
            bit enable,
            bit clear
        );
            bit signed [15:0] product;
            bit signed [20:0] sum_extended;
            
            if (clear) return 20'sh0;
            if (!enable) return prev_acc;
            
            product = weight * input_val;
            sum_extended = $signed(prev_acc) + $signed({{4{product[15]}}, product});
            
            // Saturation
            if (sum_extended > 21'sh7FFFF)
                return 20'sh7FFFF;
            else if (sum_extended < -21'sh80000)
                return 20'sh80000;
            else
                return sum_extended[19:0];
        endfunction
    endclass
    
    // Matrix Multiplication Reference Model
    class matrix_reference_model;
        static function void compute_matmul(
            input bit signed [7:0] A [0:7][0:7],
            input bit signed [7:0] B [0:7][0:7],
            output bit signed [19:0] C [0:7][0:7]
        );
            for (int i = 0; i < 8; i++) begin
                for (int j = 0; j < 8; j++) begin
                    bit signed [31:0] sum = 0;
                    for (int k = 0; k < 8; k++) begin
                        sum += $signed(A[i][k]) * $signed(B[k][j]);
                    end
                    // Saturate to 20 bits
                    if (sum > 32'sh7FFFF)
                        C[i][j] = 20'sh7FFFF;
                    else if (sum < -32'sh80000)
                        C[i][j] = 20'sh80000;
                    else
                        C[i][j] = sum[19:0];
                end
            end
        endfunction
    endclass
    
    // Activation Reference Model
    class activation_reference_model;
        static function bit signed [15:0] compute_activation(
            bit [1:0] func_select,
            bit signed [15:0] data_in
        );
            case (func_select)
                2'b00: return data_in;  // Linear
                
                2'b01: return (data_in[15] == 1'b1) ? 16'sh0000 : data_in;  // ReLU
                
                2'b10: begin  // Sigmoid
                    if (data_in < -16'sd2048) return 16'sh0000;
                    else if (data_in < -16'sd512) return 16'sh0040 + ((data_in + 16'sd512) >>> 4);
                    else if (data_in < 16'sd512) return 16'sh0080 + (data_in >>> 2);
                    else if (data_in < 16'sd2048) return 16'sh00C0 + ((data_in - 16'sd512) >>> 4);
                    else return 16'sh0100;
                end
                
                2'b11: begin  // Tanh
                    if (data_in < -16'sd1024) return -16'sh0100;
                    else if (data_in < -16'sd256) return -16'sh0100 + ((data_in + 16'sd1024) >>> 2);
                    else if (data_in < 16'sd256) return data_in;
                    else if (data_in < 16'sd1024) return 16'sh0100 - ((16'sd1024 - data_in) >>> 2);
                    else return 16'sh0100;
                end
            endcase
        endfunction
    endclass
    
    // ==========================================================================
    // Scoreboard Base Class
    // ==========================================================================
    
    class neurisc_scoreboard extends uvm_scoreboard;
        int pass_count;
        int fail_count;
        int total_tests;
        
        `uvm_component_utils(neurisc_scoreboard)
        
        function new(string name, uvm_component parent);
            super.new(name, parent);
            pass_count = 0;
            fail_count = 0;
            total_tests = 0;
        endfunction
        
        function void report_phase(uvm_phase phase);
            super.report_phase(phase);
            `uvm_info(get_type_name(), 
                $sformatf("\n========================================\n  Test Results Summary\n========================================\n  Total Tests: %0d\n  Passed: %0d\n  Failed: %0d\n  Pass Rate: %0.2f%%\n========================================",
                total_tests, pass_count, fail_count, 
                (pass_count * 100.0) / total_tests), UVM_LOW)
        endfunction
    endclass
    
    // ==========================================================================
    // Test Status Tracker
    // ==========================================================================
    
    typedef struct {
        string test_name;
        bit    passed;
        string description;
        int    errors;
    } test_result_t;
    
    class test_reporter;
        static test_result_t results[$];
        
        static function void add_result(string name, bit passed, string desc, int errors = 0);
            test_result_t result;
            result.test_name = name;
            result.passed = passed;
            result.description = desc;
            result.errors = errors;
            results.push_back(result);
        endfunction
        
        static function void print_report();
            int total = results.size();
            int passed = 0;
            
            $display("\n");
            $display("================================================================================");
            $display("                     NEURISC SOC VERIFICATION REPORT");
            $display("================================================================================");
            $display("");
            $display("%-40s  %-10s  %s", "Test Name", "Status", "Description");
            $display("--------------------------------------------------------------------------------");
            
            foreach(results[i]) begin
                string status = results[i].passed ? "PASS ✓" : "FAIL ✗";
                $display("%-40s  %-10s  %s", results[i].test_name, status, results[i].description);
                if (!results[i].passed && results[i].errors > 0)
                    $display("                                          └─ %0d errors detected", results[i].errors);
                if (results[i].passed) passed++;
            end
            
            $display("--------------------------------------------------------------------------------");
            $display("Summary: %0d/%0d tests passed (%.1f%%)", passed, total, (passed * 100.0) / total);
            $display("================================================================================");
            $display("");
        endfunction
    endclass

endpackage

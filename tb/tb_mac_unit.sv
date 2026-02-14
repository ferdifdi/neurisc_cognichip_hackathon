// =============================================================================
// Testbench: tb_mac_unit (FIXED)
// Description: Simple testbench for MAC unit verification
// =============================================================================

module tb_mac_unit;

    // Clock and reset
    logic clock;
    logic reset;
    
    // MAC unit signals
    logic        enable;
    logic        clear_acc;
    logic signed [7:0]  weight_in;
    logic signed [7:0]  input_in;
    logic signed [7:0]  weight_out;
    logic signed [7:0]  input_out;
    logic signed [19:0] accumulator;
    
    // Test variables
    int error_count;
    logic signed [19:0] expected_acc;
    
    // DUT instantiation
    mac_unit dut (
        .clock(clock),
        .reset(reset),
        .enable(enable),
        .clear_acc(clear_acc),
        .weight_in(weight_in),
        .input_in(input_in),
        .weight_out(weight_out),
        .input_out(input_out),
        .accumulator(accumulator)
    );
    
    // Clock generation (10ns period = 100MHz)
    initial begin
        clock = 0;
        forever #5 clock = ~clock;
    end
    
    // =========================================================================
    // Task: Apply one MAC operation cleanly
    //   - Sets inputs at negedge (setup time)
    //   - Enables for exactly 1 posedge
    //   - Disables before the next posedge
    // =========================================================================
    task automatic do_mac(input logic signed [7:0] w, input logic signed [7:0] i);
        @(negedge clock);       // Setup inputs before rising edge
        weight_in = w;
        input_in  = i;
        enable    = 1;
        @(posedge clock);       // MAC latches on this edge
        @(negedge clock);       // Immediately disable after
        enable = 0;
    endtask
    
    // =========================================================================
    // Task: Clear accumulator cleanly
    // =========================================================================
    task automatic do_clear();
        @(negedge clock);
        clear_acc = 1;
        @(posedge clock);
        @(negedge clock);
        clear_acc = 0;
    endtask
    
    // =========================================================================
    // Task: Wait and read result (1 cycle for output register)
    // =========================================================================
    task automatic wait_result();
        @(posedge clock);       // Wait for result to propagate
    endtask

    // Test sequence
    initial begin
        $display("TEST START");
        $display("\n========================================");
        $display("  MAC Unit Testbench");
        $display("========================================\n");
        
        error_count = 0;
        
        // Initialize signals
        reset = 1;
        enable = 0;
        clear_acc = 0;
        weight_in = 0;
        input_in = 0;
        
        // Reset sequence
        repeat(5) @(posedge clock);
        reset = 0;
        @(posedge clock);
        
        // ==================================================================
        // Test 1: Basic MAC Operation
        // ==================================================================
        $display("Test 1: Basic MAC - 5 x 3 = 15");
        do_clear();
        do_mac(8'sd5, 8'sd3);
        wait_result();
        
        expected_acc = 20'sd15;
        if (accumulator === expected_acc) begin
            $display("  PASS: Result = %0d (expected %0d)", accumulator, expected_acc);
        end else begin
            $display("  FAIL: Result = %0d (expected %0d)", accumulator, expected_acc);
            error_count++;
        end
        
        // ==================================================================
        // Test 2: Accumulation
        // ==================================================================
        $display("\nTest 2: Accumulation - 15 + (2 x 4) = 23");
        do_mac(8'sd2, 8'sd4);
        wait_result();
        
        expected_acc = 20'sd23;
        if (accumulator === expected_acc) begin
            $display("  PASS: Accumulated to %0d", accumulator);
        end else begin
            $display("  FAIL: Result = %0d (expected %0d)", accumulator, expected_acc);
            error_count++;
        end
        
        // ==================================================================
        // Test 3: Clear Accumulator
        // ==================================================================
        $display("\nTest 3: Clear Accumulator");
        do_clear();
        wait_result();
        
        if (accumulator === 20'sh0) begin
            $display("  PASS: Cleared to 0");
        end else begin
            $display("  FAIL: Result = %0d (expected 0)", accumulator);
            error_count++;
        end
        
        // ==================================================================
        // Test 4: Negative Numbers
        // ==================================================================
        $display("\nTest 4: Negative Numbers - (-6) x 7 = -42");
        do_clear();
        do_mac(-8'sd6, 8'sd7);
        wait_result();
        
        expected_acc = -20'sd42;
        if (accumulator === expected_acc) begin
            $display("  PASS: Result = %0d", $signed(accumulator));
        end else begin
            $display("  FAIL: Result = %0d (expected %0d)", $signed(accumulator), $signed(expected_acc));
            error_count++;
        end
        
        // ==================================================================
        // Test 5: Positive Overflow (Saturation)
        // ==================================================================
        $display("\nTest 5: Positive Overflow Saturation");
        do_clear();
        
        // 127 * 127 = 16129 per cycle
        // 20-bit signed max = 524287
        // 524287 / 16129 = ~32.5 cycles to overflow
        // Use 50 cycles to guarantee saturation
        @(negedge clock);
        enable = 1;
        weight_in = 8'sd127;
        input_in = 8'sd127;
        repeat(50) @(posedge clock);
        @(negedge clock);
        enable = 0;
        wait_result();
        
        if (accumulator === 20'sh7FFFF) begin
            $display("  PASS: Saturated to max (0x%0h)", accumulator);
        end else begin
            $display("  FAIL: Result = 0x%0h (expected 0x7FFFF)", accumulator);
            error_count++;
        end
        
        // ==================================================================
        // Test 6: Negative Overflow (Saturation)
        // ==================================================================
        $display("\nTest 6: Negative Overflow Saturation");
        do_clear();
        
        // -128 * 127 = -16256 per cycle
        // 20-bit signed min = -524288
        // 524288 / 16256 = ~32.3 cycles to overflow
        // Use 50 cycles to guarantee saturation
        @(negedge clock);
        enable = 1;
        weight_in = -8'sd128;
        input_in = 8'sd127;
        repeat(50) @(posedge clock);
        @(negedge clock);
        enable = 0;
        wait_result();
        
        if (accumulator === 20'sh80000) begin
            $display("  PASS: Saturated to min (0x%0h)", accumulator);
        end else begin
            $display("  FAIL: Result = 0x%0h (expected 0x80000)", accumulator);
            error_count++;
        end
        
        // ==================================================================
        // Test 7: Enable Control
        // ==================================================================
        $display("\nTest 7: Enable Control - Accumulator Holds When Disabled");
        do_clear();
        do_mac(8'sd10, 8'sd10);  // acc = 100
        wait_result();
        expected_acc = accumulator;
        
        // Disable and apply different inputs — accumulator should NOT change
        @(negedge clock);
        enable = 0;
        weight_in = 8'sd99;
        input_in = 8'sd99;
        repeat(3) @(posedge clock);
        
        if (accumulator === expected_acc) begin
            $display("  PASS: Held at %0d when disabled", accumulator);
        end else begin
            $display("  FAIL: Changed to %0d (should hold at %0d)", accumulator, expected_acc);
            error_count++;
        end
        
        // ==================================================================
        // Test 8: Pass-through Signals
        // ==================================================================
        $display("\nTest 8: Systolic Pass-through Signals");
        @(negedge clock);
        enable = 1;
        weight_in = 8'sd42;
        input_in = 8'sd73;
        @(posedge clock);       // DUT registers on this edge
        @(posedge clock);       // Output available after 1 cycle delay
        
        if (weight_out === 8'sd42 && input_out === 8'sd73) begin
            $display("  PASS: Pass-through correct (w=%0d, i=%0d)", weight_out, input_out);
        end else begin
            $display("  FAIL: Pass-through incorrect (w=%0d, i=%0d)", weight_out, input_out);
            error_count++;
        end
        
        @(negedge clock);
        enable = 0;
        
        // ==================================================================
        // Final Summary
        // ==================================================================
        $display("\n========================================");
        if (error_count == 0) begin
            $display("  ALL TESTS PASSED");
            $display("  8/8 tests successful");
            $display("TEST PASSED");
        end else begin
            $display("  TESTS FAILED");
            $display("  %0d errors detected", error_count);
            $display("TEST FAILED");
        end
        $display("========================================\n");
        
        #100;
        $finish;
    end
    
    // Waveform dump
    initial begin
        $dumpfile("dumpfile.fst");
        $dumpvars(0, tb_mac_unit);
    end
    
    // Timeout watchdog
    initial begin
        #500000;
        $display("\nERROR: Test timeout!");
        $display("TEST FAILED");
        $finish;
    end

endmodule
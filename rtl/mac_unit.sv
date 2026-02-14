// for systolic array
// =============================================================================

module mac_unit (
    input  logic        clock,
    input  logic        reset,
    input  logic        enable,
    input  logic        clear_acc,
    input  logic signed [7:0] weight_in,
    input  logic signed [7:0] input_in,
    
    output logic signed [7:0]  weight_out,
    output logic signed [7:0]  input_out,
    output logic signed [19:0] accumulator
);

    // Saturation function
    function automatic signed [19:0] saturate(input signed [31:0] val);
        if (val > 32'sd524287)
            return 20'sd524287;
        else if (val < -32'sd524288)
            return -20'sd524288;
        else
            return val[19:0];
    endfunction
    
    // Combinational logic with always_comb
    logic signed [15:0] product;
    logic signed [31:0] sum;
    logic signed [19:0] next_acc;
    
    always_comb begin
        product = weight_in * input_in;
        sum = $signed(accumulator) + $signed(product);
        if (sum > 32'sd524287)
            next_acc = 20'sd524287;
        else if (sum < -32'sd524288)
            next_acc = -20'sd524288;
        else
            next_acc = sum;  // Let SystemVerilog truncate automatically
    end
    
    // Sequential accumulator
    always_ff @(posedge clock) begin
        if (reset)
            accumulator <= 20'sh0;
        else if (clear_acc)
            accumulator <= 20'sh0;
        else if (enable)
            accumulator <= next_acc;
    end
    
    // Pass-through
    always_ff @(posedge clock) begin
        if (reset) begin
            weight_out <= 8'sh0;
            input_out  <= 8'sh0;
        end else begin
            weight_out <= weight_in;
            input_out  <= input_in;
        end
    end

endmodule

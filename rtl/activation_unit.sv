// =============================================================================
// Module: activation_unit
// Description: Configurable activation function unit for neural networks
// 
// Supported Functions:
// - Linear (passthrough): y = x
// - ReLU: y = max(0, x)
// - Sigmoid: y ≈ 1/(1+e^-x) using piecewise linear approximation
// - Tanh: y ≈ tanh(x) using piecewise linear approximation
//
// Input/Output: 16-bit signed fixed-point (Q8.8 format)
// =============================================================================

module activation_unit (
    input  logic        clock,          // Clock signal
    input  logic        reset,          // Synchronous reset (active high)
    input  logic [1:0]  func_select,    // Activation function selector
                                        // 00: Linear, 01: ReLU, 10: Sigmoid, 11: Tanh
    input  logic signed [15:0] data_in,      // 16-bit signed input
    output logic signed [15:0] data_out      // 16-bit signed output
);

    // Activation function types
    localparam [1:0] FUNC_LINEAR  = 2'b00;
    localparam [1:0] FUNC_RELU    = 2'b01;
    localparam [1:0] FUNC_SIGMOID = 2'b10;
    localparam [1:0] FUNC_TANH    = 2'b11;
    
    // Internal signals
    logic signed [15:0] linear_out;
    logic signed [15:0] relu_out;
    logic signed [15:0] sigmoid_out;
    logic signed [15:0] tanh_out;
    logic signed [15:0] activation_result;
    
    // ==========================================================================
    // Linear Activation (Passthrough)
    // ==========================================================================
    always_comb begin
        linear_out = data_in;
    end
    
    // ==========================================================================
    // ReLU Activation: max(0, x)
    // ==========================================================================
    always_comb begin
        if (data_in[15] == 1'b1) begin  // Negative (MSB is sign bit)
            relu_out = 16'sh0000;
        end else begin
            relu_out = data_in;
        end
    end
    
    // ==========================================================================
    // Sigmoid Activation: Piecewise Linear Approximation
    // σ(x) = 1/(1+e^-x)
    // 
    // Approximation regions (Q8.8 fixed-point):
    // x < -8.0  : y ≈ 0
    // -8 to -2  : y ≈ linear interpolation
    // -2 to 2   : y ≈ 0.5 + 0.25*x (main linear region)
    // 2 to 8    : y ≈ linear interpolation
    // x > 8.0   : y ≈ 1.0 (saturate)
    // ==========================================================================
    always_comb begin
        // Q8.8 format: 8 integer bits, 8 fractional bits
        // Scale factor: 256 (2^8)
        
        if (data_in < -16'sd2048) begin  // x < -8.0
            sigmoid_out = 16'sh0000;  // ≈ 0
            
        end else if (data_in < -16'sd512) begin  // -8.0 ≤ x < -2.0
            // Linear interpolation: slope ≈ 0.1
            // y ≈ 0.25 + 0.1*(x + 2)
            sigmoid_out = 16'sh0040 + ((data_in + 16'sd512) >>> 4);  // 64 + x/16
            
        end else if (data_in < 16'sd512) begin  // -2.0 ≤ x < 2.0
            // Main linear region: y ≈ 0.5 + 0.25*x
            sigmoid_out = 16'sh0080 + (data_in >>> 2);  // 128 + x/4
            
        end else if (data_in < 16'sd2048) begin  // 2.0 ≤ x < 8.0
            // Linear interpolation: slope ≈ 0.1
            // y ≈ 0.75 + 0.1*(x - 2)
            sigmoid_out = 16'sh00C0 + ((data_in - 16'sd512) >>> 4);  // 192 + x/16
            
        end else begin  // x ≥ 8.0
            sigmoid_out = 16'sh0100;  // ≈ 1.0 (saturate)
        end
    end
    
    // ==========================================================================
    // Tanh Activation: Piecewise Linear Approximation
    // tanh(x) = (e^x - e^-x)/(e^x + e^-x)
    // 
    // Approximation regions (Q8.8 fixed-point):
    // x < -4.0  : y ≈ -1.0
    // -4 to -1  : y ≈ linear interpolation
    // -1 to 1   : y ≈ x (main linear region)
    // 1 to 4    : y ≈ linear interpolation
    // x > 4.0   : y ≈ 1.0
    // ==========================================================================
    always_comb begin
        // Q8.8 format: 8 integer bits, 8 fractional bits
        
        if (data_in < -16'sd1024) begin  // x < -4.0
            tanh_out = -16'sh0100;  // ≈ -1.0
            
        end else if (data_in < -16'sd256) begin  // -4.0 ≤ x < -1.0
            // Linear interpolation: slope ≈ 0.25
            // y ≈ -1.0 + 0.25*(x + 4)
            tanh_out = -16'sh0100 + ((data_in + 16'sd1024) >>> 2);
            
        end else if (data_in < 16'sd256) begin  // -1.0 ≤ x < 1.0
            // Main linear region: y ≈ x
            tanh_out = data_in;
            
        end else if (data_in < 16'sd1024) begin  // 1.0 ≤ x < 4.0
            // Linear interpolation: slope ≈ 0.25
            // y ≈ 1.0 - 0.25*(4 - x)
            tanh_out = 16'sh0100 - ((16'sd1024 - data_in) >>> 2);
            
        end else begin  // x ≥ 4.0
            tanh_out = 16'sh0100;  // ≈ 1.0
        end
    end
    
    // ==========================================================================
    // Function Selector Multiplexer
    // ==========================================================================
    always_comb begin
        case (func_select)
            FUNC_LINEAR:  activation_result = linear_out;
            FUNC_RELU:    activation_result = relu_out;
            FUNC_SIGMOID: activation_result = sigmoid_out;
            FUNC_TANH:    activation_result = tanh_out;
            default:      activation_result = linear_out;
        endcase
    end
    
    // ==========================================================================
    // Output Register (Optional pipelining)
    // ==========================================================================
    always_ff @(posedge clock) begin
        if (reset) begin
            data_out <= 16'sh0000;
        end else begin
            data_out <= activation_result;
        end
    end

endmodule

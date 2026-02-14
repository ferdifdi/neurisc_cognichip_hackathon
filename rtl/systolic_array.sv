// =============================================================================
// Module: systolic_array
// Description: Parameterized Systolic Array for Matrix Multiplication
// 
// Architecture:
// - SIZE x SIZE MAC units arranged in a 2D grid (default 8x8)
// - Weights flow horizontally (left to right)
// - Activations flow vertically (top to bottom)
// - Implements matrix multiplication: C = A x B
// - Supports accumulate mode for back-to-back K-tile processing
//
// Parameters:
// - SIZE: Array dimension (default 8, set to 16 for high-performance)
//
// Timing:
// - Total computation cycles = 3*SIZE - 1
// - Accumulate mode: skip clear between K-tiles for pipelined operation
// =============================================================================

module systolic_array #(
    parameter int SIZE = 8
) (
    input  logic        clock,              // Clock signal
    input  logic        reset,              // Synchronous reset (active high)
    input  logic        start,              // Start computation
    input  logic        accumulate,         // Accumulate mode (don't clear between tiles)
    
    // Weight inputs (one per row, fed sequentially)
    input  logic signed [7:0] weight_in [0:SIZE-1],
    
    // Activation inputs (one per column, fed sequentially)
    input  logic signed [7:0] input_in [0:SIZE-1],
    
    // Output accumulator results (SIZE x SIZE matrix)
    output logic signed [19:0] result [0:SIZE-1][0:SIZE-1],
    
    // Status signals
    output logic        done,               // Computation complete
    output logic [7:0]  cycle_count         // Cycle counter
);

    // Internal wire arrays for systolic connections
    logic signed [7:0] weight_horizontal [0:SIZE-1][0:SIZE];  // Horizontal weight flow
    logic signed [7:0] input_vertical [0:SIZE][0:SIZE-1];     // Vertical input flow
    
    // MAC unit control signals
    logic mac_enable;
    logic mac_clear;
    
    // State machine for control
    typedef enum logic [2:0] {
        IDLE,
        LOADING,
        COMPUTING,
        DONE_STATE
    } state_t;
    
    state_t current_state, next_state;
    
    // Cycle counter for tracking computation progress
    logic [7:0] cycle_counter;
    logic       count_enable;
    
    // Total cycles needed: 2*SIZE-1 for data propagation + SIZE for accumulation
    localparam int TOTAL_CYCLES = 3 * SIZE - 1;
    
    // ==========================================================================
    // Instantiate 8x8 grid of MAC units
    // ==========================================================================
    generate
        genvar row, col;
        for (row = 0; row < SIZE; row++) begin : gen_rows
            for (col = 0; col < SIZE; col++) begin : gen_cols
                mac_unit mac_pe (
                    .clock       (clock),
                    .reset       (reset),
                    .enable      (mac_enable),
                    .clear_acc   (mac_clear),
                    .weight_in   (weight_horizontal[row][col]),
                    .input_in    (input_vertical[row][col]),
                    .weight_out  (weight_horizontal[row][col+1]),
                    .input_out   (input_vertical[row+1][col]),
                    .accumulator (result[row][col])
                );
            end
        end
    endgenerate
    
    // ==========================================================================
    // Connect array boundaries to external inputs
    // ==========================================================================
    
    // Left edge: Connect weight inputs to first column
    generate
        genvar wi;
        for (wi = 0; wi < SIZE; wi++) begin : gen_weight_in
            assign weight_horizontal[wi][0] = weight_in[wi];
        end
    endgenerate
    
    // Top edge: Connect activation inputs to first row
    generate
        genvar ii;
        for (ii = 0; ii < SIZE; ii++) begin : gen_input_in
            assign input_vertical[0][ii] = input_in[ii];
        end
    endgenerate
    
    // ==========================================================================
    // Control State Machine
    // ==========================================================================
    
    // State register
    always_ff @(posedge clock) begin
        if (reset) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (start) begin
                    next_state = LOADING;
                end
            end
            
            LOADING: begin
                // Transition to computing after first cycle
                next_state = COMPUTING;
            end
            
            COMPUTING: begin
                if (cycle_counter >= TOTAL_CYCLES - 1) begin
                    next_state = DONE_STATE;
                end
            end
            
            DONE_STATE: begin
                if (!start) begin
                    next_state = IDLE;
                end
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    // Output logic
    always_comb begin
        mac_enable = 1'b0;
        mac_clear  = 1'b0;
        count_enable = 1'b0;
        done = 1'b0;
        
        case (current_state)
            IDLE: begin
                mac_clear = 1'b1;  // Clear accumulators when idle
            end
            
            LOADING: begin
                mac_clear = !accumulate;  // Clear only if not accumulating across K-tiles
                count_enable = 1'b1;
            end
            
            COMPUTING: begin
                mac_enable = 1'b1;  // Enable accumulation
                count_enable = 1'b1;
            end
            
            DONE_STATE: begin
                done = 1'b1;  // Signal completion
            end
        endcase
    end
    
    // ==========================================================================
    // Cycle Counter
    // ==========================================================================
    
    always_ff @(posedge clock) begin
        if (reset || current_state == IDLE) begin
            cycle_counter <= 8'd0;
        end else if (count_enable) begin
            cycle_counter <= cycle_counter + 1'b1;
        end
    end
    
    // Export cycle count
    assign cycle_count = cycle_counter;

endmodule

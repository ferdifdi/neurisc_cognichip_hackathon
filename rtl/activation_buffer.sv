// =============================================================================
// Module: activation_buffer
// Description: 128KB Double-Buffered SRAM for Neural Network Activations
// 
// Specifications:
// - Total capacity: 128KB (131,072 bytes)
// - Organization: Two banks of 64KB each (8K entries × 64 bits per bank)
// - Double buffering: One bank for read, one for write (ping-pong operation)
// - Bank swapping: Automatic or manual swap between read/write banks
// - Enables overlap of computation and data loading
// =============================================================================

module activation_buffer (
    input  logic        clock,              // Clock signal
    input  logic        reset,              // Synchronous reset (active high)
    
    // Control signals
    input  logic        swap_banks,         // Swap read/write banks (pulse)
    output logic        active_bank,        // Current active bank (0 or 1)
    
    // Read interface (from active read bank)
    input  logic        read_enable,        // Read enable
    input  logic [12:0] read_addr,          // Read address (13 bits = 8K locations)
    output logic [63:0] read_data,          // Read data (registered)
    
    // Write interface (to active write bank)
    input  logic        write_enable,       // Write enable
    input  logic [12:0] write_addr,         // Write address (13 bits = 8K locations)
    input  logic [63:0] write_data          // Write data
);

    // Memory arrays: Two banks of 8K entries × 64 bits each
    // Bank 0: 8192 * 64 bits = 524,288 bits = 65,536 bytes = 64KB
    // Bank 1: 8192 * 64 bits = 524,288 bits = 65,536 bytes = 64KB
    // Total: 128KB
    logic [63:0] bank0 [0:8191];
    logic [63:0] bank1 [0:8191];
    
    // Bank selection control
    // active_bank = 0: Read from bank0, Write to bank1
    // active_bank = 1: Read from bank1, Write to bank0
    logic current_read_bank;
    logic current_write_bank;
    
    // Internal read data from each bank
    logic [63:0] bank0_read_data;
    logic [63:0] bank1_read_data;
    
    // ==========================================================================
    // Bank Swap Control
    // ==========================================================================
    always_ff @(posedge clock) begin
        if (reset) begin
            current_read_bank <= 1'b0;   // Initially read from bank 0
            current_write_bank <= 1'b1;  // Initially write to bank 1
        end else if (swap_banks) begin
            // Swap banks: toggle read/write bank assignment
            current_read_bank <= ~current_read_bank;
            current_write_bank <= ~current_write_bank;
        end
    end
    
    assign active_bank = current_read_bank;
    
    // ==========================================================================
    // Bank 0: Read/Write Logic
    // ==========================================================================
    always_ff @(posedge clock) begin
        if (reset) begin
            bank0_read_data <= 64'h0;
        end else begin
            // Write to bank 0 when it's the write bank
            if (write_enable && (current_write_bank == 1'b0)) begin
                bank0[write_addr] <= write_data;
            end
            
            // Read from bank 0 when it's the read bank
            if (read_enable && (current_read_bank == 1'b0)) begin
                bank0_read_data <= bank0[read_addr];
            end
        end
    end
    
    // ==========================================================================
    // Bank 1: Read/Write Logic
    // ==========================================================================
    always_ff @(posedge clock) begin
        if (reset) begin
            bank1_read_data <= 64'h0;
        end else begin
            // Write to bank 1 when it's the write bank
            if (write_enable && (current_write_bank == 1'b1)) begin
                bank1[write_addr] <= write_data;
            end
            
            // Read from bank 1 when it's the read bank
            if (read_enable && (current_read_bank == 1'b1)) begin
                bank1_read_data <= bank1[read_addr];
            end
        end
    end
    
    // ==========================================================================
    // Output Multiplexer: Select data from active read bank
    // ==========================================================================
    always_ff @(posedge clock) begin
        if (reset) begin
            read_data <= 64'h0;
        end else begin
            read_data <= (current_read_bank == 1'b0) ? bank0_read_data : bank1_read_data;
        end
    end

endmodule

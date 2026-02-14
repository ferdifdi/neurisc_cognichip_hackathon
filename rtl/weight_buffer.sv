// =============================================================================
// Module: weight_buffer
// Description: 256KB Dual-Port SRAM for Neural Network Weight Storage
// 
// Specifications:
// - Total capacity: 256KB (262,144 bytes)
// - Organization: 32K entries × 64 bits (8 bytes per entry)
// - Dual-port: Simultaneous independent read/write on separate ports
// - Port A: Read/Write port
// - Port B: Read/Write port
// - Synchronous read/write with registered outputs
// =============================================================================

module weight_buffer (
    input  logic        clock,              // Clock signal
    input  logic        reset,              // Synchronous reset (active high)
    
    // Port A interface
    input  logic        port_a_enable,      // Port A enable
    input  logic        port_a_write_en,    // Port A write enable (1=write, 0=read)
    input  logic [14:0] port_a_addr,        // Port A address (15 bits = 32K locations)
    input  logic [63:0] port_a_write_data,  // Port A write data (64 bits)
    output logic [63:0] port_a_read_data,   // Port A read data (registered)
    
    // Port B interface
    input  logic        port_b_enable,      // Port B enable
    input  logic        port_b_write_en,    // Port B write enable (1=write, 0=read)
    input  logic [14:0] port_b_addr,        // Port B address (15 bits = 32K locations)
    input  logic [63:0] port_b_write_data,  // Port B write data (64 bits)
    output logic [63:0] port_b_read_data    // Port B read data (registered)
);

    // Memory array: 32K entries of 64 bits each
    // Total size: 32768 * 64 bits = 2,097,152 bits = 262,144 bytes = 256KB
    logic [63:0] memory [0:32767];
    
    // ==========================================================================
    // Port A: Read/Write Logic
    // ==========================================================================
    always_ff @(posedge clock) begin
        if (reset) begin
            port_a_read_data <= 64'h0;
        end else if (port_a_enable) begin
            if (port_a_write_en) begin
                // Write operation
                memory[port_a_addr] <= port_a_write_data;
                // Optional: read-during-write returns old data
                port_a_read_data <= memory[port_a_addr];
            end else begin
                // Read operation
                port_a_read_data <= memory[port_a_addr];
            end
        end
    end
    
    // ==========================================================================
    // Port B: Read/Write Logic
    // ==========================================================================
    always_ff @(posedge clock) begin
        if (reset) begin
            port_b_read_data <= 64'h0;
        end else if (port_b_enable) begin
            if (port_b_write_en) begin
                // Write operation
                memory[port_b_addr] <= port_b_write_data;
                // Optional: read-during-write returns old data
                port_b_read_data <= memory[port_b_addr];
            end else begin
                // Read operation
                port_b_read_data <= memory[port_b_addr];
            end
        end
    end
    
    // Note: In case of simultaneous writes to the same address from both ports,
    // Port B write takes priority (last write wins). This is typical SRAM behavior.

endmodule

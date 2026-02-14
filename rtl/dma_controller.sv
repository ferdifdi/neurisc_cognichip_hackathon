// =============================================================================
// Module: dma_controller
// Description: DMA Controller with 2D Strided Access Support
// 
// Features:
// - Configurable source and destination addresses
// - Programmable transfer size
// - 2D addressing with row/column stride support
// - Burst transfer capability
// - Busy/Done status flags
// - Supports both linear and strided memory access patterns
//
// Use Cases:
// - Linear memory copy: Set row_count=1, transfer_size=total_bytes
// - 2D tensor copy: Configure row_count, col_count, src_stride, dst_stride
// =============================================================================

module dma_controller (
    input  logic        clock,              // Clock signal
    input  logic        reset,              // Synchronous reset (active high)
    
    // Control interface
    input  logic        start,              // Start DMA transfer (pulse)
    output logic        busy,               // Transfer in progress
    output logic        done,               // Transfer complete (pulse)
    
    // Configuration registers (written before starting transfer)
    input  logic [31:0] src_base_addr,      // Source base address
    input  logic [31:0] dst_base_addr,      // Destination base address
    input  logic [15:0] transfer_size,      // Transfer size per row (bytes)
    
    // 2D addressing configuration
    input  logic        mode_2d,            // 0=linear, 1=2D strided
    input  logic [15:0] row_count,          // Number of rows (for 2D mode)
    input  logic [15:0] col_count,          // Number of columns per row
    input  logic [15:0] src_row_stride,     // Source stride between rows (bytes)
    input  logic [15:0] dst_row_stride,     // Destination stride between rows (bytes)
    
    // Memory interface (simplified - can be AXI4/AHB in real design)
    output logic        mem_read_en,        // Memory read enable
    output logic        mem_write_en,       // Memory write enable
    output logic [31:0] mem_read_addr,      // Memory read address
    output logic [31:0] mem_write_addr,     // Memory write address
    input  logic [63:0] mem_read_data,      // Memory read data
    output logic [63:0] mem_write_data,     // Memory write data
    input  logic        mem_ready           // Memory ready (for handshaking)
);

    // State machine
    typedef enum logic [2:0] {
        IDLE,
        READ_DATA,
        WAIT_READ,
        WRITE_DATA,
        WAIT_WRITE,
        ROW_DONE,
        TRANSFER_DONE
    } state_t;
    
    state_t current_state, next_state;
    
    // Internal registers
    logic [31:0] current_src_addr;
    logic [31:0] current_dst_addr;
    logic [15:0] bytes_transferred;
    logic [15:0] current_row;
    logic [15:0] current_col;
    logic [63:0] data_buffer;
    
    // ==========================================================================
    // State Machine - Sequential Logic
    // ==========================================================================
    always_ff @(posedge clock) begin
        if (reset) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end
    
    // ==========================================================================
    // State Machine - Next State Logic
    // ==========================================================================
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (start) begin
                    next_state = READ_DATA;
                end
            end
            
            READ_DATA: begin
                next_state = WAIT_READ;
            end
            
            WAIT_READ: begin
                if (mem_ready) begin
                    next_state = WRITE_DATA;
                end
            end
            
            WRITE_DATA: begin
                next_state = WAIT_WRITE;
            end
            
            WAIT_WRITE: begin
                if (mem_ready) begin
                    // Check if row is complete
                    if (mode_2d) begin
                        if (current_col >= col_count - 1) begin
                            next_state = ROW_DONE;
                        end else begin
                            next_state = READ_DATA;
                        end
                    end else begin
                        // Linear mode
                        if (bytes_transferred >= transfer_size - 8) begin
                            next_state = TRANSFER_DONE;
                        end else begin
                            next_state = READ_DATA;
                        end
                    end
                end
            end
            
            ROW_DONE: begin
                // Check if all rows are complete
                if (current_row >= row_count - 1) begin
                    next_state = TRANSFER_DONE;
                end else begin
                    next_state = READ_DATA;
                end
            end
            
            TRANSFER_DONE: begin
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    // ==========================================================================
    // Address and Counter Management
    // ==========================================================================
    always_ff @(posedge clock) begin
        if (reset) begin
            current_src_addr <= 32'h0;
            current_dst_addr <= 32'h0;
            bytes_transferred <= 16'h0;
            current_row <= 16'h0;
            current_col <= 16'h0;
            data_buffer <= 64'h0;
            
        end else begin
            case (current_state)
                IDLE: begin
                    if (start) begin
                        // Initialize transfer
                        current_src_addr <= src_base_addr;
                        current_dst_addr <= dst_base_addr;
                        bytes_transferred <= 16'h0;
                        current_row <= 16'h0;
                        current_col <= 16'h0;
                    end
                end
                
                WAIT_READ: begin
                    if (mem_ready) begin
                        // Capture read data
                        data_buffer <= mem_read_data;
                    end
                end
                
                WAIT_WRITE: begin
                    if (mem_ready) begin
                        if (mode_2d) begin
                            // 2D mode: increment column
                            current_col <= current_col + 1;
                            current_src_addr <= current_src_addr + 8;  // 8 bytes per transfer
                            current_dst_addr <= current_dst_addr + 8;
                        end else begin
                            // Linear mode: increment by 8 bytes
                            bytes_transferred <= bytes_transferred + 8;
                            current_src_addr <= current_src_addr + 8;
                            current_dst_addr <= current_dst_addr + 8;
                        end
                    end
                end
                
                ROW_DONE: begin
                    // Move to next row with stride
                    current_row <= current_row + 1;
                    current_col <= 16'h0;
                    // Apply row stride
                    current_src_addr <= src_base_addr + ((current_row + 1) * src_row_stride);
                    current_dst_addr <= dst_base_addr + ((current_row + 1) * dst_row_stride);
                end
            endcase
        end
    end
    
    // ==========================================================================
    // Output Logic
    // ==========================================================================
    always_comb begin
        // Default values
        busy = 1'b0;
        done = 1'b0;
        mem_read_en = 1'b0;
        mem_write_en = 1'b0;
        mem_read_addr = 32'h0;
        mem_write_addr = 32'h0;
        mem_write_data = 64'h0;
        
        case (current_state)
            IDLE: begin
                busy = 1'b0;
            end
            
            READ_DATA: begin
                busy = 1'b1;
                mem_read_en = 1'b1;
                mem_read_addr = current_src_addr;
            end
            
            WAIT_READ: begin
                busy = 1'b1;
            end
            
            WRITE_DATA: begin
                busy = 1'b1;
                mem_write_en = 1'b1;
                mem_write_addr = current_dst_addr;
                mem_write_data = data_buffer;
            end
            
            WAIT_WRITE: begin
                busy = 1'b1;
            end
            
            ROW_DONE: begin
                busy = 1'b1;
            end
            
            TRANSFER_DONE: begin
                done = 1'b1;
            end
        endcase
    end

endmodule

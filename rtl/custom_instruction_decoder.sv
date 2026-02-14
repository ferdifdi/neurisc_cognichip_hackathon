// =============================================================================
// Module: custom_instruction_decoder
// Description: Custom Instruction Decoder for RISC-V NPU Extension
// 
// Custom Instructions (simplified representation):
// - NPU_MATMUL: Trigger matrix multiplication on systolic array
// - NPU_LOAD:   Load data into weight/activation buffers
// - NPU_STORE:  Store results from NPU
// - NPU_CONFIG: Configure NPU parameters
// - DMA_START:  Start DMA transfer
// - DMA_CONFIG: Configure DMA parameters
//
// This is a simplified interface - in a real implementation, this would
// decode custom opcodes from the RISC-V instruction stream.
// =============================================================================

module custom_instruction_decoder (
    input  logic        clock,              // Clock signal
    input  logic        reset,              // Synchronous reset (active high)
    
    // Memory-mapped register interface (from external bus/CPU)
    input  logic        reg_write_en,       // Register write enable
    input  logic        reg_read_en,        // Register read enable
    input  logic [31:0] reg_addr,           // Register address
    input  logic [31:0] reg_write_data,     // Register write data
    output logic [31:0] reg_read_data,      // Register read data
    output logic        reg_ready,          // Register access ready
    
    // NPU Control Interface (to NPU subsystem)
    output logic        npu_start,          // Start NPU computation
    output logic        npu_clear,          // Clear NPU accumulators
    input  logic        npu_done,           // NPU computation done
    input  logic        npu_busy,           // NPU busy status
    input  logic [31:0] npu_cycle_count,    // NPU performance counter
    
    // DMA Control Interface (to DMA controller)
    output logic        dma_start,          // Start DMA transfer
    output logic [31:0] dma_src_addr,       // DMA source address
    output logic [31:0] dma_dst_addr,       // DMA destination address
    output logic [15:0] dma_size,           // DMA transfer size
    output logic        dma_mode_2d,        // DMA 2D mode enable
    output logic [15:0] dma_row_count,      // DMA row count
    output logic [15:0] dma_col_count,      // DMA column count
    output logic [15:0] dma_src_stride,     // DMA source stride
    output logic [15:0] dma_dst_stride,     // DMA destination stride
    input  logic        dma_busy,           // DMA busy status
    input  logic        dma_done,           // DMA done status
    
    // Activation Function Control
    output logic [1:0]  activation_select   // Activation function selector
);

    // Memory-mapped register addresses
    localparam logic [31:0] ADDR_NPU_CTRL       = 32'h80002000;  // NPU control register
    localparam logic [31:0] ADDR_NPU_STATUS     = 32'h80002004;  // NPU status register
    localparam logic [31:0] ADDR_NPU_CYCLES     = 32'h80002008;  // NPU cycle counter
    localparam logic [31:0] ADDR_NPU_ACTIVATION = 32'h8000200C;  // Activation function select
    
    localparam logic [31:0] ADDR_DMA_CTRL       = 32'h80001000;  // DMA control register
    localparam logic [31:0] ADDR_DMA_STATUS     = 32'h80001004;  // DMA status register
    localparam logic [31:0] ADDR_DMA_SRC        = 32'h80001008;  // DMA source address
    localparam logic [31:0] ADDR_DMA_DST        = 32'h8000100C;  // DMA destination address
    localparam logic [31:0] ADDR_DMA_SIZE       = 32'h80001010;  // DMA transfer size
    localparam logic [31:0] ADDR_DMA_MODE       = 32'h80001014;  // DMA mode config
    localparam logic [31:0] ADDR_DMA_ROWS       = 32'h80001018;  // DMA row count
    localparam logic [31:0] ADDR_DMA_COLS       = 32'h8000101C;  // DMA column count
    localparam logic [31:0] ADDR_DMA_SRC_STRIDE = 32'h80001020;  // DMA source stride
    localparam logic [31:0] ADDR_DMA_DST_STRIDE = 32'h80001024;  // DMA dest stride
    
    // Internal control registers
    logic [31:0] npu_ctrl_reg;
    logic [31:0] dma_ctrl_reg;
    logic [31:0] dma_src_reg;
    logic [31:0] dma_dst_reg;
    logic [31:0] dma_size_reg;
    logic [31:0] dma_mode_reg;
    logic [31:0] dma_rows_reg;
    logic [31:0] dma_cols_reg;
    logic [31:0] dma_src_stride_reg;
    logic [31:0] dma_dst_stride_reg;
    logic [1:0]  activation_reg;
    
    // Control bit positions
    localparam int NPU_CTRL_START_BIT = 0;
    localparam int NPU_CTRL_CLEAR_BIT = 1;
    localparam int DMA_CTRL_START_BIT = 0;
    
    // ==========================================================================
    // Register Write Logic
    // ==========================================================================
    always_ff @(posedge clock) begin
        if (reset) begin
            npu_ctrl_reg <= 32'h0;
            dma_ctrl_reg <= 32'h0;
            dma_src_reg <= 32'h0;
            dma_dst_reg <= 32'h0;
            dma_size_reg <= 32'h0;
            dma_mode_reg <= 32'h0;
            dma_rows_reg <= 32'h0;
            dma_cols_reg <= 32'h0;
            dma_src_stride_reg <= 32'h0;
            dma_dst_stride_reg <= 32'h0;
            activation_reg <= 2'b00;
            
        end else if (reg_write_en) begin
            case (reg_addr)
                ADDR_NPU_CTRL:       npu_ctrl_reg <= reg_write_data;
                ADDR_NPU_ACTIVATION: activation_reg <= reg_write_data[1:0];
                ADDR_DMA_CTRL:       dma_ctrl_reg <= reg_write_data;
                ADDR_DMA_SRC:        dma_src_reg <= reg_write_data;
                ADDR_DMA_DST:        dma_dst_reg <= reg_write_data;
                ADDR_DMA_SIZE:       dma_size_reg <= reg_write_data;
                ADDR_DMA_MODE:       dma_mode_reg <= reg_write_data;
                ADDR_DMA_ROWS:       dma_rows_reg <= reg_write_data;
                ADDR_DMA_COLS:       dma_cols_reg <= reg_write_data;
                ADDR_DMA_SRC_STRIDE: dma_src_stride_reg <= reg_write_data;
                ADDR_DMA_DST_STRIDE: dma_dst_stride_reg <= reg_write_data;
            endcase
        end else begin
            // Auto-clear start bits after one cycle
            npu_ctrl_reg[NPU_CTRL_START_BIT] <= 1'b0;
            npu_ctrl_reg[NPU_CTRL_CLEAR_BIT] <= 1'b0;
            dma_ctrl_reg[DMA_CTRL_START_BIT] <= 1'b0;
        end
    end
    
    // ==========================================================================
    // Register Read Logic
    // ==========================================================================
    always_comb begin
        reg_read_data = 32'h0;
        reg_ready = 1'b1;
        
        if (reg_read_en) begin
            case (reg_addr)
                ADDR_NPU_CTRL:       reg_read_data = npu_ctrl_reg;
                ADDR_NPU_STATUS:     reg_read_data = {30'h0, npu_busy, npu_done};
                ADDR_NPU_CYCLES:     reg_read_data = npu_cycle_count;
                ADDR_NPU_ACTIVATION: reg_read_data = {30'h0, activation_reg};
                ADDR_DMA_CTRL:       reg_read_data = dma_ctrl_reg;
                ADDR_DMA_STATUS:     reg_read_data = {30'h0, dma_busy, dma_done};
                ADDR_DMA_SRC:        reg_read_data = dma_src_reg;
                ADDR_DMA_DST:        reg_read_data = dma_dst_reg;
                ADDR_DMA_SIZE:       reg_read_data = dma_size_reg;
                ADDR_DMA_MODE:       reg_read_data = dma_mode_reg;
                ADDR_DMA_ROWS:       reg_read_data = dma_rows_reg;
                ADDR_DMA_COLS:       reg_read_data = dma_cols_reg;
                ADDR_DMA_SRC_STRIDE: reg_read_data = dma_src_stride_reg;
                ADDR_DMA_DST_STRIDE: reg_read_data = dma_dst_stride_reg;
                default:             reg_read_data = 32'hDEADBEEF;
            endcase
        end
    end
    
    // ==========================================================================
    // Output Assignments
    // ==========================================================================
    assign npu_start = npu_ctrl_reg[NPU_CTRL_START_BIT];
    assign npu_clear = npu_ctrl_reg[NPU_CTRL_CLEAR_BIT];
    
    assign dma_start = dma_ctrl_reg[DMA_CTRL_START_BIT];
    assign dma_src_addr = dma_src_reg;
    assign dma_dst_addr = dma_dst_reg;
    assign dma_size = dma_size_reg[15:0];
    assign dma_mode_2d = dma_mode_reg[0];
    assign dma_row_count = dma_rows_reg[15:0];
    assign dma_col_count = dma_cols_reg[15:0];
    assign dma_src_stride = dma_src_stride_reg[15:0];
    assign dma_dst_stride = dma_dst_stride_reg[15:0];
    
    assign activation_select = activation_reg;

endmodule
